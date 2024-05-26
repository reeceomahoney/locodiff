import logging
import math
import os
from collections import deque
from faulthandler import disable
from functools import partial

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm, trange
import wandb

import locodiff.utils as utils

# A logger for this file
log = logging.getLogger(__name__)


class Agent:

    def __init__(
        self,
        model: DictConfig,
        optimization: DictConfig,
        dataset_fn: DictConfig,
        device: str,
        max_train_steps: int,
        eval_every_n_steps: int,
        use_ema: bool,
        num_sampling_steps: int,
        lr_scheduler: DictConfig,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        decay: float,
        update_ema_every_n_steps: int,
        T: int,
        T_cond: int,
        T_action: int,
        obs_dim: int,
        pred_obs_dim: int,
        action_dim: int,
        num_envs: int,
        sim_every_n_steps: int,
        weight_decay: float,
        cond_lambda: int,
        cond_mask_prob: float,
    ):
        # model
        self.model = hydra.utils.instantiate(model).to(device)
        self.T = T  # set to 0 to just predict the next action
        self.T_cond = T_cond
        self.T_action = T_action
        self.obs_context = deque(maxlen=self.T_cond)
        self.action_context = deque(maxlen=self.T_cond - 1)

        total_params = sum(p.numel() for p in self.model.get_params())
        log.info("Parameter count: {:e}".format(total_params))

        # training
        optim_groups = self.model.inner_model.get_optim_groups(weight_decay)
        self.optimizer = hydra.utils.instantiate(optimization, optim_groups)
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler, optimizer=self.optimizer
        )
        self.steps = 0
        self.max_train_steps = int(max_train_steps)
        self.eval_every_n_steps = eval_every_n_steps

        # ema
        self.ema_helper = utils.ExponentialMovingAverage(
            self.model.get_params(), decay, device
        )
        self.use_ema = use_ema
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps

        # diffusion
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_lambda = cond_lambda
        self.cond_mask_prob = cond_mask_prob

        # env
        self.obs_dim = obs_dim
        self.pred_obs_dim = pred_obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.sim_every_n_steps = sim_every_n_steps

        self.train_loader, self.test_loader, self.scaler = hydra.utils.instantiate(
            dataset_fn
        )

        # Update scaler to shifted position scale
        batch = next(iter(self.train_loader))
        self.scaler.update_pos_scale(batch, self.T_cond)

        # misc
        self.device = device
        self.env = None
        self.working_dir = None
        self.total_mse = None

    def train_agent(self):
        """
        Main training loop
        """
        best_test_mse = 1e10
        generator = iter(self.train_loader)

        for step in tqdm(
            range(self.max_train_steps), position=0, leave=True, dynamic_ncols=True
        ):
            # evaluate
            if not self.steps % self.eval_every_n_steps:
                log_info = {
                    "total_mse": [],
                    "first_mse": [],
                    "last_mse": [],
                    "state_mse": [],
                    "action_mse": [],
                }
                for batch in tqdm(
                    self.test_loader, desc="Evaluating", position=0, leave=True
                ):
                    info = self.evaluate(batch)
                    for key in log_info:
                        log_info[key].append(info[key])
                for key in log_info:
                    log_info[key] = sum(log_info[key]) / len(log_info[key])
                if log_info["total_mse"] < best_test_mse:
                    best_test_mse = log_info["total_mse"]
                    self.store_model_weights(self.working_dir)
                    log.info("New best test loss. Stored weights have been updated!")
                log_info["lr"] = self.optimizer.param_groups[0]["lr"]

                wandb.log({k: v for k, v in log_info.items()}, step=self.steps)

            # train
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(self.train_loader)
                batch_loss = self.train_step(next(generator))
            if not self.steps % 100:
                wandb.log({"loss": batch_loss}, step=self.steps)

            # simulate
            if not self.steps % self.sim_every_n_steps:
                results = self.env.simulate(self)
                wandb.log(results, step=self.steps)

        self.store_model_weights(self.working_dir)
        log.info("Training done!")

    def train_step(self, batch: dict):
        state_in, sa_out, goal = self.process_batch(batch)

        self.model.train()
        self.model.training = True

        noise = torch.randn_like(sa_out)
        sigma = self.make_sample_density()(shape=(len(sa_out),), device=self.device)
        loss = self.model.loss(sa_out, state_in, noise, sigma, goal=goal)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.steps += 1

        # update the ema model
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict, **kwargs):
        """
        Evaluates the model using the provided batch of data and returns the mean squared error (MSE) loss.
        """
        state_in, sa_out, goal = self.process_batch(batch)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        self.model.training = False

        # get the sigma distribution for sampling based on Karras et al. 2022
        sigmas = utils.get_sigmas_exponential(
            self.num_sampling_steps, self.sigma_min, self.sigma_max, self.device
        )
        x = torch.randn_like(sa_out) * self.sigma_max
        x_0, epsilon = self.sample_ddim(x, sigmas, state_in, goal, **kwargs)

        mse = nn.functional.mse_loss(x_0, sa_out, reduction="none")
        total_mse = mse.mean().item()
        self.total_mse = total_mse

        # state and action mse
        state_mse = mse[:, :, : self.pred_obs_dim].mean().item()
        action_mse = mse[:, :, self.pred_obs_dim :].mean().item()

        # mse of the first and last timestep
        first_mse = mse[:, 0, :].mean().item()
        last_mse = mse[:, -1, :].mean().item()
        timestep_mse = mse.mean(dim=(0, 2))

        prediction = self.scaler.inverse_scale_output(x_0)[..., : self.pred_obs_dim]

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        info = {
            "mse": mse,
            "total_mse": total_mse,
            "state_mse": state_mse,
            "action_mse": action_mse,
            "first_mse": first_mse,
            "last_mse": last_mse,
            "timestep_mse": timestep_mse,
            "prediction": prediction,
            "goal": goal,
        }
        if "num_steps" in kwargs:
            info["inputs"] = (epsilon, state_in, goal)
        return info

    def reset(self):
        """Resets the context of the model."""
        self.obs_context.clear()
        self.action_context.clear()

    @torch.no_grad()
    def predict(self, batch: dict, new_sampling_steps=None) -> torch.Tensor:
        """
        Predicts the output of the model based on the provided batch of data.
        """
        batch["observation"] = self.stack_context(batch["observation"])
        state_in, _, goal = self.process_batch(batch)

        if new_sampling_steps is not None:
            n_sampling_steps = new_sampling_steps
        else:
            n_sampling_steps = self.num_sampling_steps

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()

        # get the sigma distribution for the desired sampling method
        sigmas = utils.get_sigmas_exponential(
            n_sampling_steps, self.sigma_min, self.sigma_max, self.device
        )

        sa_dim = self.pred_obs_dim + self.action_dim
        x = torch.randn((self.num_envs, self.T, sa_dim), device=self.device)
        x *= self.sigma_max

        x_0, denoised = self.sample_ddim(x, sigmas, state_in, goal, predict=True)

        # get the action for the current timestep
        x_0 = self.scaler.clip(x_0)
        pred_traj = self.scaler.inverse_scale_output(x_0).cpu().numpy()
        pred_action = pred_traj[:, : self.T_action, self.pred_obs_dim :].copy()

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        return pred_action, pred_traj

    def stack_context(self, state):
        """
        Helper function to handle obs history
        """
        self.obs_context.append(state)
        while len(self.obs_context) < self.T_cond:
            self.obs_context.append(state)
        return torch.stack(tuple(self.obs_context), dim=1)

    @torch.no_grad()
    def sample_ddim(self, x_t, sigmas, cond, goal, **kwargs):
        """
        Sample from the model using the DDIM sampler

        Args:
            x_t (torch.Tensor): The initial state-action noise tensor.
            goals (torch.Tensor): One-hot encoding of the goals.
            sigmas (torch.Tensor): The sigma distribution for the sampling.
            cond (torch.Tensor): The conditioning input for the model.
        Returns:
            torch.Tensor: The predicted output of the model.
        """
        s_in = x_t.new_ones([x_t.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        num_steps = kwargs.get("num_steps", len(sigmas) - 1)
        kwargs["goal"] = goal
        predict = kwargs.get("predict", False)

        for i in trange(num_steps, disable=disable):
            if predict:
                denoised = self.cfg_forward(x_t, cond, sigmas[i] * s_in, **kwargs)
            else:
                denoised = self.model(x_t, cond, sigmas[i] * s_in, **kwargs)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            x_t = (sigma_fn(t_next) / sigma_fn(t)) * x_t - (-h).expm1() * denoised

        return x_t, denoised

    def cfg_forward(self, x_t, cond, sigma, **kwargs):
        """
        Classifier-free guidance sample
        """
        out = self.model(x_t, cond, sigma, **kwargs)

        if self.cond_mask_prob > 0:
            kwargs["uncond"] = True
            out_uncond = self.model(x_t, cond, sigma, **kwargs)
            out = out_uncond + self.cond_lambda * (out - out_uncond)

        return out

    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
        self.model.load_state_dict(
            torch.load(
                os.path.join(weights_path, "model_state_dict.pth"),
                map_location=self.device,
            ),
            strict=False,
        )

        # Load scaler attributes
        scaler_state = torch.load(
            os.path.join(weights_path, "scaler.pth"), map_location=self.device
        )
        self.scaler.x_max = scaler_state["x_max"]
        self.scaler.x_min = scaler_state["x_min"]
        self.scaler.y_max = scaler_state["y_max"]
        self.scaler.y_min = scaler_state["y_min"]
        self.scaler.goal_max = scaler_state["goal_max"]
        self.scaler.goal_min = scaler_state["goal_min"]

        log.info("Loaded pre-trained model parameters and scaler")

    def store_model_weights(self, store_path: str) -> None:
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        torch.save(
            self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth")
        )
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(
            self.model.state_dict(),
            os.path.join(store_path, "non_ema_model_state_dict.pth"),
        )

        # Save scaler attributes
        torch.save(
            {
                "x_max": self.scaler.x_max,
                "x_min": self.scaler.x_min,
                "y_max": self.scaler.y_max,
                "y_min": self.scaler.y_min,
                "goal_max": self.scaler.goal_max,
                "goal_min": self.scaler.goal_min,
            },
            os.path.join(store_path, "scaler.pth"),
        )

    @torch.no_grad()
    def make_sample_density(self):
        """
        Generate a density function for training sigmas
        """
        sd_config = []
        loc = sd_config["loc"] if "loc" in sd_config else math.log(self.sigma_data)
        scale = sd_config["scale"] if "scale" in sd_config else 0.5
        min_value = (
            sd_config["min_value"] if "min_value" in sd_config else self.sigma_min
        )
        max_value = (
            sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
        )
        return partial(
            utils.rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )

    @torch.no_grad()
    def process_batch(self, batch: dict):
        """
        Processes a batch of data and returns the state, action and goal
        """
        state = self.get_to_device(batch, "observation")
        action = self.get_to_device(batch, "action")
        goal = self.get_to_device(batch, "goal")

        # Centre posisition around the current state
        current_pos = state[:, self.T_cond - 1, :2].clone()
        state[..., :2] = state[..., :2] - current_pos.unsqueeze(1)
        state_in = self.scaler.scale_input(state[:, : self.T_cond])

        # Action
        if action is not None:
            sa = torch.cat([state[..., : self.pred_obs_dim], action], dim=-1)
            sa_out = self.scaler.scale_output(sa[:, self.T_cond - 1 :])
        else:
            sa_out = None

        goal[..., :2] -= current_pos
        goal = self.scaler.scale_goal(goal)
        # goal[..., -1] = 0

        return state_in, sa_out, goal

    def get_to_device(self, batch, key):
        return (
            batch.get(key).clone().to(self.device)
            if batch.get(key) is not None
            else None
        )
