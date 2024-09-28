import logging
import os
import random
import sys
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import locodiff.utils as utils
import wandb
from locodiff.agent import Agent

# A logger for this file
log = logging.getLogger(__name__)


class ClassifierWorkspace:

    def __init__(
        self,
        agent: Agent,
        classifier: nn.Module,
        optimizer: Callable,
        lr_scheduler: Callable,
        dataset_fn: Tuple[DataLoader, DataLoader, utils.Scaler],
        agent_path: str,
        wandb_project: str,
        wandb_mode: str,
        train_steps: int,
        eval_every: int,
        seed: int,
        device: str,
        obs_dim: int,
        action_dim: int,
        skill_dim: int,
        T: int,
        T_cond: int,
        T_action: int,
        num_envs: int,
        sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        return_horizon: int,
        reward_fn: str,
    ):
        # debug mode
        if sys.gettrace() is not None:
            self.output_dir = "/tmp"
            wandb_mode = "disabled"
        else:
            self.output_dir = HydraConfig.get().runtime.output_dir
            wandb_mode = "online"

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # training
        self.train_steps = int(train_steps)
        self.eval_every = int(eval_every)
        self.device = device

        # dataloader and scaler
        self.train_loader, self.test_loader, self.scaler = dataset_fn

        # agent
        self.agent = agent
        self.agent.load_state_dict(
            torch.load(
                os.path.join(agent_path, "model", "model.pt"), map_location=self.device
            ),
            strict=False,
        )
        self.agent.sampling_steps = sampling_steps

        # classifier
        self.classifier = classifier

        # optimizer and lr scheduler
        optim_groups = self.classifier.get_optim_groups()
        self.optimizer = optimizer(optim_groups)
        self.lr_scheduler = lr_scheduler(self.optimizer)

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
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

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
        self.eval_keys = ["mse"]

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
                if log_info["mse"] < best_total_mse:
                    best_total_mse = log_info["mse"]
                    self.save()
                    log.info("New best test loss. Stored weights have been updated!")

                # log to wandb
                wandb.log({k: v for k, v in log_info.items()}, step=step)

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
        num_steps = random.randint(0, self.sampling_steps - 1)
        x_0 = self.agent(data_dict, num_steps=num_steps)

        # calculate loss
        pred_return = self.classifier(x_0, data_dict)
        loss = nn.functional.mse_loss(pred_return, data_dict["return"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict:
        data_dict = self.process_batch(batch)
        num_steps = random.randint(0, self.sampling_steps - 1)
        x_0 = self.agent(data_dict, num_steps=num_steps)

        pred_return = self.classifier(x_0, data_dict)
        mse = nn.functional.mse_loss(pred_return, data_dict["return"], reduction="none")
        return {"mse": mse.mean().item()}

    ######################
    # Saving and Loading #
    ######################

    def load(self, weights_path: str) -> None:
        model_dir = os.path.join(weights_path, "model")

        self.classifier.load_state_dict(
            torch.load(
                os.path.join(model_dir, "classifier.pt"), map_location=self.device
            ),
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
        torch.save(
            self.classifier.state_dict(), os.path.join(model_dir, "classifier.pt")
        )

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
                        raw_obs[:, self.T_cond - 1 : self.T_cond + self.T - 1],
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
        vel_cmd = torch.randint(0, 2, (batch_size, 1), device=self.device).float()
        return vel_cmd * 2 - 1

    def compute_returns(self, obs, vel_cmd):
        rewards = utils.reward_function(obs, vel_cmd, self.reward_fn)
        rewards = rewards[:, self.T_cond - 1 :] - 1

        gammas = torch.tensor([0.99**i for i in range(self.return_horizon)]).to(
            self.device
        )
        returns = torch.zeros((obs.shape[0], self.T)).to(self.device)
        for i in range(self.T):
            ret = (rewards[:, i : i + self.return_horizon] * gammas).sum(dim=-1)
            ret = torch.exp(ret / 10)
            ret = (ret - ret.min()) / (ret.max() - ret.min())
            returns[:, i] = ret

        return returns.unsqueeze(-1)
