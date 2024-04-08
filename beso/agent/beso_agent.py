import logging
import math
import os
from collections import deque
from faulthandler import disable
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from tqdm import tqdm, trange

import beso.agent.utils as utils
from beso.agent.foot_grid import FootGrid
from beso.agent.classifier_free_sampler import (
    ClassifierFreeSampleModel,
)
from beso.agent.gc_sampling import get_sigmas_exponential
from beso.networks.ema_helper.ema import ExponentialMovingAverage
from beso.networks.scaler.scaler_class import Scaler

# A logger for this file
log = logging.getLogger(__name__)


class BesoAgent:

    def __init__(
        self,
        model: DictConfig,
        optimization: DictConfig,
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
    ):
        # model
        self.model = hydra.utils.instantiate(model).to(device)
        # set to 0 to just predict the next action
        self.T = T
        self.T_cond = T_cond
        self.T_action = T_action
        self.obs_context = deque(maxlen=self.T_cond)
        self.action_context = deque(maxlen=self.T_cond - 1)
        total_params = sum(p.numel() for p in self.model.get_params())
        log.info("The model has a total amount of {:e} parameters".format(total_params))

        # training
        optim_groups = self.model.inner_model.get_optim_groups()
        self.optimizer = hydra.utils.instantiate(optimization, optim_groups)
        self.lr_scheduler = hydra.utils.instantiate(
            lr_scheduler, optimizer=self.optimizer
        )
        self.steps = 0
        self.max_train_steps = int(max_train_steps)
        self.eval_every_n_steps = eval_every_n_steps

        # ema
        self.ema_helper = ExponentialMovingAverage(
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

        # env
        self.obs_dim = obs_dim
        self.pred_obs_dim = pred_obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.sim_every_n_steps = sim_every_n_steps

        # misc
        self.scaler = None
        self.working_dir = os.getcwd()
        self.device = device

    def train_agent(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        eval_fn,
    ):
        """
        Main training loop
        """
        best_test_mse = 1e10
        generator = iter(train_loader)

        for step in tqdm(range(self.max_train_steps), position=0, leave=True):
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
                    test_loader, desc="Evaluating", position=0, leave=True
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

                wandb.log({k: v for k, v in log_info.items()})

            # train
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(next(generator))
            wandb.log({"loss": batch_loss})

            # simulate
            if not self.steps % self.sim_every_n_steps:
                results = eval_fn(self)
                wandb.log(results)

        self.store_model_weights(self.working_dir)
        log.info("Training done!")

    def train_step(self, batch: dict):
        state, action, _ = self.process_batch(batch)
        state_action = torch.cat([state, action], dim=-1)

        self.model.train()
        self.model.training = True

        sa_dim = self.pred_obs_dim + self.action_dim
        noise = torch.randn((state.shape[0], self.T + 1, sa_dim)).to(self.device)
        sigma = self.make_sample_density()(shape=(len(action),), device=self.device)
        loss = self.model.loss(state_action, noise, sigma)

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
    def evaluate(self, batch: dict):
        """
        Evaluates the model using the provided batch of data and returns the mean squared error (MSE) loss.
        """
        state, action, _ = self.process_batch(batch)

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()
        self.model.training = False

        # get the sigma distribution for sampling based on Karras et al. 2022
        sigmas = get_sigmas_exponential(
            self.num_sampling_steps, self.sigma_min, self.sigma_max, self.device
        )

        sa_dim = self.pred_obs_dim + self.action_dim
        x = torch.randn((state.shape[0], self.T + 1, sa_dim)) * self.sigma_max
        x = x.to(self.device)

        # generate the action based on the chosen sampler type
        state_action = torch.cat([state, action], dim=-1)
        cond = state_action[:, : self.T_cond, : self.obs_dim]
        x_0 = self.sample_ddim(x, sigmas, cond)

        mse = nn.functional.mse_loss(
            x_0, state_action[:, self.T_cond - 1 :, :], reduction="none"
        )
        total_mse = mse.mean().item()

        # state and action mse
        state_mse = mse[:, :, : self.pred_obs_dim].mean().item()
        action_mse = mse[:, :, self.pred_obs_dim :].mean().item()

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
            "state_mse": state_mse,
            "action_mse": action_mse,
            "first_mse": first_mse,
            "last_mse": last_mse,
            "timestep_mse": timestep_mse,
        }
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
        state, _, _ = self.process_batch(batch)

        input_state = self.stack_context(state)

        if new_sampling_steps is not None:
            n_sampling_steps = new_sampling_steps
        else:
            n_sampling_steps = self.num_sampling_steps

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()

        # get the sigma distribution for the desired sampling method
        sigmas = get_sigmas_exponential(
            n_sampling_steps, self.sigma_min, self.sigma_max, self.device
        )

        sa_dim = self.pred_obs_dim + self.action_dim
        x = torch.randn((self.num_envs, self.T + 1, sa_dim), device=self.device)
        x *= self.sigma_max

        cond = input_state
        x_0 = self.sample_ddim(x, sigmas, cond)

        # get the action for the current timestep
        x_0 = x_0[:, : self.T_action, self.pred_obs_dim :]
        x_0 = self.scaler.clip_action(x_0)

        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        model_pred = self.scaler.inverse_scale_output(x_0)
        # self.action_context.append(x_0)
        return model_pred

    def stack_context(self, state):
        """
        Helper function to handle obs and action history
        """
        self.obs_context.append(state)
        while len(self.obs_context) < self.T_cond:
            self.obs_context.append(state)
        input_state = torch.stack(tuple(self.obs_context), dim=1)

        # pad = torch.zeros(state.shape[0], self.action_dim).to(self.device)
        # while len(self.action_context) < self.T_cond - 1:
        #     self.action_context.append(pad)
        # input_action = torch.stack(
        #     [*tuple(self.action_context), pad],
        #     dim=1,
        # )

        return input_state

    @torch.no_grad()
    def sample_ddim(self, x_t, sigmas, cond):
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

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = self.model(x_t, cond, sigmas[i] * s_in)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            x_t = (sigma_fn(t_next) / sigma_fn(t)) * x_t - (-h).expm1() * denoised

        return x_t

    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
        self.model.load_state_dict(
            torch.load(os.path.join(weights_path, "non_ema_model_state_dict.pth")),
            strict=False,
        )
        self.ema_helper = ExponentialMovingAverage(
            self.model.get_params(), self.decay, self.device
        )
        log.info("Loaded pre-trained model parameters")

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
        get_to_device = lambda key: (
            batch.get(key).to(self.device) if batch.get(key) is not None else None
        )

        state = get_to_device("observation")
        state = self.scaler.scale_input(state)

        action = get_to_device("action")
        if action is not None:
            action = self.scaler.scale_output(action)

        goal = get_to_device("goal")

        return state, action, goal

    def get_scaler(self, scaler: Scaler):
        self.scaler = scaler

    def get_constraints(self, state: torch.Tensor) -> torch.Tensor:
        """
        Method to calculate the constraints for the given state

        Returns:
            constraints: (B, 1, 1)
        """
        # calculate active foot grids
        future_states = state[:, self.T_cond :, :]
        future_states = self.scaler.inverse_scale_input(future_states)
        active_grids = self.foot_grid.get_active_grids(future_states)

        constraints = active_grids.reshape(state.shape[0], -1)
        constraints = constraints.to(torch.float32).unsqueeze(1)
        return constraints

    def calculate_constraint_vector(
        self, state: torch.Tensor, threshold: float, greater_than: bool = 0
    ) -> torch.Tensor:
        if greater_than:
            constraints = torch.where(state.norm(dim=-1).max(-1)[0] > threshold, 1, 0)
        else:
            constraints = torch.where(state.norm(dim=-1).max(-1)[0] < threshold, 1, 0)
        constraints = constraints.to(torch.float32).view(-1, 1, 1)
        return constraints
