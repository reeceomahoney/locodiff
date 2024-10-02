import logging
import os
import random
import sys
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import locodiff.utils as utils
from env.env import RaisimEnv

# A logger for this file
log = logging.getLogger(__name__)


class Workspace:

    def __init__(
        self,
        model,
        wrapper: Callable,
        agent: Callable,
        optimizer: Callable,
        lr_scheduler: Callable,
        dataset_fn: Tuple[DataLoader, DataLoader, utils.Scaler],
        env: RaisimEnv,
        ema_helper: Callable,
        wandb_project: str,
        wandb_mode: str,
        train_steps: int,
        eval_every: int,
        sim_every: int,
        seed: int,
        device: str,
        use_ema: bool,
        obs_dim: int,
        action_dim: int,
        skill_dim: int,
        T: int,
        T_cond: int,
        T_action: int,
        num_envs: int,
        sampling_steps: int,
        cond_mask_prob: float,
        return_horizon: int,
        reward_fn: str,
    ):
        # debug mode
        if sys.gettrace() is not None:
            self.output_dir = "/tmp"
            sim_every = 10
        else:
            self.output_dir = HydraConfig.get().runtime.output_dir

        # agent
        self.agent = agent(model=wrapper(model=model))

        # optimizer and lr scheduler
        optim_groups = self.agent.get_optim_groups()
        self.optimizer = optimizer(optim_groups)
        self.lr_scheduler = lr_scheduler(self.optimizer)

        # dataloader and scaler
        self.train_loader, self.test_loader, self.scaler = dataset_fn

        # env
        self.env = env

        # ema
        self.ema_helper = ema_helper(self.agent.get_params())
        self.use_ema = use_ema

        # training
        self.train_steps = int(train_steps)
        self.eval_every = int(eval_every)
        self.sim_every = int(sim_every)
        self.device = device

        # dims
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.T = T
        self.T_cond = T_cond
        self.T_action = T_action

        self.obs_hist = torch.zeros((num_envs, T_cond, obs_dim), device=device)
        self.skill_hist = torch.zeros((num_envs, T_cond, skill_dim), device=device)

        # diffusion
        self.sampling_steps = sampling_steps
        self.cond_mask_prob = cond_mask_prob

        # reward
        self.return_horizon = return_horizon
        self.reward_fn = reward_fn

        # logging
        os.makedirs(self.output_dir + "/model", exist_ok=True)
        wandb.init(
            project=wandb_project,
            mode=wandb_mode,
            dir=self.output_dir,
            config=wandb.config,
        )
        self.eval_keys = ["total_mse", "first_mse", "last_mse"]
        if self.cond_mask_prob > 0:
            self.eval_keys.append("output_divergence")

    ############
    # Training #
    ############

    def train(self):
        """
        Main training loop
        """
        best_total_mse = 1e10
        generator = iter(self.train_loader)

        for step in trange(self.train_steps, desc="Training", dynamic_ncols=True):
            # evaluate
            if not step % self.eval_every:
                # reset the log_info
                log_info = {k: [] for k in self.eval_keys}

                # run evaluation
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    info = self.evaluate(batch)
                    log_info = {k: v + [info[k]] for k, v in log_info.items()}

                # calculate the means and lr
                log_info = {k: sum(v) / len(v) for k, v in log_info.items()}
                log_info["lr"] = self.optimizer.param_groups[0]["lr"]

                # save model if it has improved mse
                if log_info["total_mse"] < best_total_mse:
                    best_total_mse = log_info["total_mse"]
                    self.save()
                    log.info("New best test loss. Stored weights have been updated!")

                # log to wandb
                wandb.log({k: v for k, v in log_info.items()}, step=step)

            # simulate
            if not step % self.sim_every:
                results = self.env.simulate(self)
                wandb.log(results, step=step)

            # train
            try:
                batch_loss = self.train_step(next(generator))
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(self.train_loader)
                batch_loss = self.train_step(next(generator))
            if not step % 100:
                wandb.log({"loss": batch_loss}, step=step)

        self.save()
        log.info("Training done!")

    def train_step(self, batch: dict):
        data_dict = self.process_batch(batch)
        loss = self.agent.loss(data_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.ema_helper.update(self.agent.parameters())
        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict:
        info = {}
        data_dict = self.process_batch(batch)

        if self.use_ema:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())

        if self.cond_mask_prob > 0:
            x_0, x_0_max_return = self.agent(data_dict)
            x_0 = self.scaler.clip(x_0)
            x_0 = self.scaler.inverse_scale_output(x_0)

            # divergence between normal and max_return outputs
            x_0_max_return = self.scaler.clip(x_0_max_return)
            x_0_max_return = self.scaler.inverse_scale_output(x_0_max_return)
            output_divergence = torch.abs(x_0 - x_0_max_return).mean().item()
            info["output_divergence"] = output_divergence
        else:
            x_0 = self.agent(data_dict)
            x_0 = self.scaler.clip(x_0)
            x_0 = self.scaler.inverse_scale_output(x_0)

        # calculate the MSE
        raw_action = self.scaler.inverse_scale_output(data_dict["action"])
        mse = nn.functional.mse_loss(x_0, raw_action, reduction="none")
        total_mse = mse.mean().item()
        first_mse = mse[:, 0, :].mean().item()
        last_mse = mse[:, -1, :].mean().item()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.agent.parameters())

        info["total_mse"] = total_mse
        info["first_mse"] = first_mse
        info["last_mse"] = last_mse

        return info

    ##############
    # Inference #
    ##############

    @torch.no_grad()
    def __call__(self, batch: dict, new_sampling_steps=None):
        batch = self.update_history(batch)
        data_dict = self.process_batch(batch)

        if new_sampling_steps is not None:
            self.agent.sampling_steps = new_sampling_steps

        if self.use_ema:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())

        if self.cond_mask_prob > 0:
            pred_action, _ = self.agent(data_dict)
        else:
            pred_action = self.agent(data_dict)

        pred_action = self.scaler.clip(pred_action)
        pred_action = self.scaler.inverse_scale_output(pred_action)
        pred_action = pred_action.cpu().numpy()
        pred_action = pred_action[:, : self.T_action].copy()

        if self.use_ema:
            self.ema_helper.restore(self.agent.parameters())

        return pred_action

    def reset(self, done):
        self.obs_hist[done] = 0
        self.skill_hist[done] = 0

    ######################
    # Saving and Loading #
    ######################

    def load(self, weights_path: str) -> None:
        model_dir = os.path.join(weights_path, "model")

        self.agent.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device),
            strict=False,
        )

        # Load scaler attributes
        scaler_state = torch.load(
            os.path.join(model_dir, "scaler.pt"), map_location=self.device
        )
        for attr, value in scaler_state.items():
            setattr(self.scaler, attr, value)

        log.info("Loaded pre-trained agent parameters and scaler")

    def save(self) -> None:
        model_dir = os.path.join(self.output_dir, "model")

        if self.use_ema:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())

        torch.save(self.agent.state_dict(), os.path.join(model_dir, "model.pt"))

        if self.use_ema:
            self.ema_helper.restore(self.agent.parameters())

        # Save scaler attributes
        scaler_state = ["x_max", "x_min", "y_max", "y_min", "x_mean", "x_std", "y_mean"]
        torch.save(
            {k: getattr(self.scaler, k) for k in scaler_state}, model_dir + "/scaler.pt"
        )

    ###################
    # Data Processing #
    ###################

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
                torch.cat(
                    [
                        # raw_obs[:, self.T_cond - 1 : self.T_cond + self.T - 1],
                        raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1],
                    ],
                    dim=-1,
                )
            )

        processed_batch = {
            "obs": obs,
            "action": action,
            "vel_cmd": vel_cmd,
            "skill": skill,
            "return": returns,
        }

        return processed_batch

    def update_history(self, batch):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = batch["obs"]
        batch["obs"] = self.obs_hist.clone()
        return batch

    def dict_to_device(self, batch):
        return {k: v.clone().to(self.device) for k, v in batch.items()}

    def sample_vel_cmd(self, batch_size):
        # vel_cmd = torch.randint(0, 2, (batch_size, 1), device=self.device).float()
        # return vel_cmd * 2 - 1
        vel_limits = [0.8, 0.5, 1.0]
        vel_cmd = torch.rand(batch_size, 3, device=self.device)

        for i in range(3):
            vel_cmd[i] = vel_cmd[i] * 2 * vel_limits[i] - vel_limits[i]

        return vel_cmd

    def compute_returns(self, obs, vel_cmd):
        rewards = utils.reward_function(obs, vel_cmd, self.reward_fn)
        rewards = (
            rewards[:, self.T_cond - 1 : self.T_cond + self.return_horizon - 1] - 1
        )

        gammas = torch.tensor([0.99**i for i in range(self.return_horizon)]).to(
            self.device
        )
        returns = (rewards * gammas).sum(dim=-1)
        returns = torch.exp(returns / 10)
        returns = (returns - returns.min()) / (returns.max() - returns.min())

        return returns.unsqueeze(-1)
