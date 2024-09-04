import logging
import math
import os

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

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
        action_dim: int,
        skill_dim: int,
        num_envs: int,
        sim_every_n_steps: int,
        weight_decay: float,
        cond_lambda: int,
        cond_mask_prob: float,
        evaluating: bool,
        reward_fn: str,
        ddpm: bool = False,
        noise_schedule: DictConfig = None,
    ):
        self.ddpm = ddpm

        # model
        self.model = hydra.utils.instantiate(model).to(device)
        self.T = T
        self.T_cond = T_cond
        self.T_action = T_action
        self.obs_hist = torch.zeros((num_envs, T_cond, obs_dim), device=device)
        self.skill_hist = torch.zeros((num_envs, T_cond, skill_dim), device=device)

        total_params = sum(p.numel() for p in self.model.get_params())
        log.info("Parameter count: {:e}".format(total_params))

        # training
        if self.ddpm:
            optim_groups = self.model.get_optim_groups(weight_decay)
            self.noise_scheduler = hydra.utils.instantiate(noise_schedule)
        else:
            optim_groups = self.model.inner_model.get_optim_groups(weight_decay)
        self.optimizer = hydra.utils.instantiate(optimization, optim_groups)
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler, optimizer=self.optimizer
        )
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
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.sim_every_n_steps = sim_every_n_steps

        if evaluating:
            # Load a dummy scaler for evaluation
            x_data = torch.zeros((1, obs_dim), device=device)
            y_data = torch.zeros((1, action_dim), device=device)
            self.scaler = utils.MinMaxScaler(x_data, y_data, device)
        else:
            self.train_loader, self.test_loader, self.scaler = hydra.utils.instantiate(
                dataset_fn
            )

        # misc
        self.device = device
        self.env = None
        self.working_dir = None
        self.total_mse = None
        self.reward_fn = reward_fn

    def train_agent(self):
        """
        Main training loop
        """
        best_test_mse = 1e10
        best_reward = -1e10
        generator = iter(self.train_loader)

        for step in tqdm(range(self.max_train_steps), dynamic_ncols=True):
            # evaluate
            if not step % self.eval_every_n_steps:
                log_info = {"total_mse": [], "first_mse": [], "last_mse": []}
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

                wandb.log({k: v for k, v in log_info.items()}, step=step)

            # simulate
            if not step % self.sim_every_n_steps:
                results = self.env.simulate(self)
                wandb.log(results, step=step)

                # save the best model by reward
                rewards = [v for k, v in results.items() if k.endswith("/reward_mean")]
                max_reward = max(rewards)
                if max_reward > best_reward:
                    best_reward = max_reward
                    self.store_model_weights(self.working_dir, best_reward=True)
                    log.info("New best reward. Stored weights have been updated!")

            # train
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(self.train_loader)
                batch_loss = self.train_step(next(generator))
            if not step % 100:
                wandb.log({"loss": batch_loss}, step=step)

        self.store_model_weights(self.working_dir)
        log.info("Training done!")

    def train_step(self, batch: dict):
        data_dict = self.process_batch(batch)

        self.model.train()
        self.model.training = True

        action = data_dict["action"]
        noise = torch.randn_like(action)
        if self.ddpm:
            timesteps = torch.randint(0, self.num_sampling_steps, (noise.shape[0],))
            noise_trajectory = self.noise_scheduler.add_noise(action, noise, timesteps)
            timesteps = timesteps.float().to(self.device)
            pred = self.model(noise_trajectory, timesteps, data_dict)
            loss = torch.nn.functional.mse_loss(pred, noise)
        else:
            sigma = self.make_sample_density(len(noise))
            pred = self.model(noise, data_dict)
            loss = self.model.loss(noise, sigma, data_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.ema_helper.update(self.model.parameters())
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict:
        """
        Calculate model prediction error
        """
        data_dict = self.process_batch(batch)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        self.model.training = False

        # get the sigma distribution for sampling based on Karras et al. 2022
        noise = torch.randn_like(data_dict["action"])
        if self.ddpm:
            self.noise_scheduler.set_timesteps(self.num_sampling_steps)
            x_0 = self.sample_ddpm(noise, data_dict)
        else:
            noise = noise * self.sigma_max
            sigmas = utils.get_sigmas_exponential(
                self.num_sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
            x_0 = self.sample_ddim(noise, sigmas, data_dict, predict=False)

        mse = nn.functional.mse_loss(x_0, data_dict["action"], reduction="none")
        total_mse = mse.mean().item()
        self.total_mse = total_mse

        # mse of the first and last timestep
        first_mse = mse[:, 0, :].mean().item()
        last_mse = mse[:, -1, :].mean().item()
        timestep_mse = mse.mean(dim=(0, 2))

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        info = {
            "mse": mse,
            "total_mse": total_mse,
            "first_mse": first_mse,
            "last_mse": last_mse,
            "timestep_mse": timestep_mse,
        }

        return info

    def reset(self, done):
        self.obs_hist[done] = 0
        self.skill_hist[done] = 0

    @torch.no_grad()
    def predict(self, batch: dict, new_sampling_steps=None):
        """
        Inference method
        """
        batch = self.stack_context(batch)
        data_dict = self.process_batch(batch)

        if new_sampling_steps is not None:
            n_sampling_steps = new_sampling_steps
        else:
            n_sampling_steps = self.num_sampling_steps

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()

        # get the sigma distribution for the desired sampling method
        noise = torch.randn(
            (self.num_envs, self.T, self.action_dim), device=self.device
        )
        if self.ddpm:
            self.noise_scheduler.set_timesteps(n_sampling_steps)
            x_0 = self.sample_ddpm(noise, data_dict, predict=True)
        else:
            noise *= self.sigma_max
            sigmas = utils.get_sigmas_exponential(
                n_sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
            x_0 = self.sample_ddim(noise, sigmas, data_dict, predict=True)

        # get the action for the current timestep
        x_0 = self.scaler.clip(x_0)
        pred_action = self.scaler.inverse_scale_output(x_0).cpu().numpy()
        pred_action = pred_action[:, : self.T_action].copy()

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        return pred_action

    def stack_context(self, batch):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = batch["obs"]
        batch["obs"] = self.obs_hist.clone()
        return batch

    @torch.no_grad()
    def sample_ddim(
        self, noise: torch.Tensor, sigmas: torch.Tensor, data_dict: dict, predict: bool
    ):
        """
        Perform inference using the DDIM sampler
        """
        x_t = noise
        s_in = x_t.new_ones([x_t.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        for i in range(len(sigmas) - 1):
            if predict:
                denoised = self.cfg_forward(x_t, sigmas[i] * s_in, data_dict)
            else:
                denoised = self.model(x_t, sigmas[i] * s_in, data_dict)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            x_t = (sigma_fn(t_next) / sigma_fn(t)) * x_t - (-h).expm1() * denoised

        return x_t

    @torch.no_grad()
    def sample_ddpm(self, noise: torch.Tensor, data_dict: dict, predict: bool = False):
        """
        Perform inference using the DDIM sampler
        """
        x_t = noise

        for t in self.noise_scheduler.timesteps:
            t_pt = t.float().to(self.device)
            if predict:
                output = self.cfg_forward(x_t, t_pt.expand(x_t.shape[0]), data_dict)
            else:
                output = self.model(x_t, t_pt.expand(x_t.shape[0]), data_dict)
            x_t = self.noise_scheduler.step(output, t, x_t).prev_sample

        return x_t

    def cfg_forward(self, x_t: torch.Tensor, sigma: torch.Tensor, data_dict: dict):
        """
        Classifier-free guidance sample
        """
        # TODO: parallelize this

        out = self.model(x_t, sigma, data_dict)

        if self.cond_mask_prob > 0:
            out_uncond = self.model(x_t, sigma, data_dict, uncond=True)
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

        log.info("Loaded pre-trained model parameters and scaler")

    def store_model_weights(self, store_path: str, best_reward: bool = False) -> None:
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        name = (
            "model_state_dict.pth" if not best_reward else "best_model_state_dict.pth"
        )
        torch.save(self.model.state_dict(), os.path.join(store_path, name))
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(
            self.model.state_dict(),
            os.path.join(store_path, "non_ema_" + name),
        )

        # Save scaler attributes
        torch.save(
            {
                "x_max": self.scaler.x_max,
                "x_min": self.scaler.x_min,
                "y_max": self.scaler.y_max,
                "y_min": self.scaler.y_min,
            },
            os.path.join(store_path, "scaler.pth"),
        )

    @torch.no_grad()
    def make_sample_density(self, size):
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
        density = utils.rand_log_logistic(
            (size,), loc, scale, min_value, max_value, self.device
        )
        return density

    @torch.no_grad()
    def process_batch(self, batch: dict) -> dict:
        batch = self.dict_to_device(batch)

        raw_obs = batch["obs"]
        raw_action = batch.get("action", None)
        skill = batch["skill"]
        vel_cmd = batch.get("vel_cmd", None)
        if vel_cmd is None:
            vel_cmd = self.sample_vel_cmd(raw_obs.shape[0])
    
        returns = batch.get("return", None)
        if returns is None:
            returns = self.compute_returns(raw_obs, vel_cmd)

        obs = self.scaler.scale_input(raw_obs[:, : self.T_cond])

        if raw_action is None:
            action = None
        else:
            action = self.scaler.scale_output(
                raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1],
            )

        processed_batch = {
            "obs": obs,
            "action": action,
            "vel_cmd": vel_cmd,
            "skill": skill,
            "return": returns,
        }

        return processed_batch

    def dict_to_device(self, batch):
        return {k: v.clone().to(self.device) for k, v in batch.items()}

    def sample_vel_cmd(self, batch_size):
        vel_cmd = torch.randint(0, 2, (batch_size, 1), device=self.device).float()
        return vel_cmd
    
    def compute_returns(self, obs, vel_cmd):
        rewards = utils.reward_function(obs, vel_cmd, self.reward_fn)
        rewards = rewards[:, self.T_cond - 1:] - 1

        horizon = 50
        gammas = torch.tensor([0.99**i for i in range(horizon)]).to(self.device)
        returns = (rewards * gammas).sum(dim=-1)
        returns = torch.exp(returns / 10)
        returns += (1 - returns.max())

        # import matplotlib.pyplot as plt

        # plt.hist(returns.flatten().cpu().numpy(), bins=20)
        # plt.xlabel("Returns")
        # plt.ylabel("Frequency")
        # plt.savefig("returns.png")
        # exit()

        return returns.unsqueeze(-1)
