import logging
import math
import os
import random
import sys
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

import locodiff.utils as utils
from env.env import RaisimEnv
from locodiff.transformer import DiffusionTransformer
from locodiff.wrapper import ScalingWrapper

# A logger for this file
log = logging.getLogger(__name__)


class Workspace:

    def __init__(
        self,
        model: DiffusionTransformer,
        wrapper: ScalingWrapper,
        optimizer: Callable,
        lr_scheduler: Callable,
        dataset_fn: Tuple[DataLoader, DataLoader, utils.Scaler],
        env: RaisimEnv,
        ema_helper: Callable,
        noise_scheduler: DDPMScheduler,
        wandb_project: str,
        train_steps: int,
        eval_every: int,
        sim_every: int,
        seed: int,
        device: str,
        use_ema: bool,
        use_ddpm: bool,
        obs_dim: int,
        action_dim: int,
        skill_dim: int,
        T: int,
        T_cond: int,
        T_action: int,
        sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        cond_lambda: int,
        cond_mask_prob: float,
        return_horizon: int,
        reward_fn: str,
    ):
        # debug mode
        if sys.gettrace() is not None:
            self.output_dir = "/tmp"
            wandb_mode = "disabled"
            sim_every = 10
        else:
            self.output_dir = HydraConfig.get().runtime.output_dir
            wandb_mode = "online"

        wandb_mode = "disabled"
        self.output_dir = "/tmp"

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # model
        self.model = wrapper(model=model)

        # optimizer and lr scheduler
        optim_groups = self.model.get_optim_groups()
        self.optimizer = optimizer(optim_groups)
        self.lr_scheduler = lr_scheduler(self.optimizer)

        # dataloader and scaler
        self.train_loader, self.test_loader, self.scaler = dataset_fn

        # env
        self.env = env
        self.num_envs = env.num_envs

        # ema
        self.ema_helper = ema_helper(self.model.get_params())
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

        self.obs_hist = torch.zeros((self.num_envs, T_cond, obs_dim), device=device)
        self.skill_hist = torch.zeros((self.num_envs, T_cond, skill_dim), device=device)

        # diffusion
        self.sampling_steps = sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_lambda = cond_lambda
        self.cond_mask_prob = cond_mask_prob

        # reward
        self.return_horizon = return_horizon
        self.reward_fn = reward_fn

        # ddpm
        self.use_ddpm = use_ddpm
        self.noise_scheduler = noise_scheduler

        # logging
        wandb.init(project=wandb_project, mode=wandb_mode, dir=self.output_dir)

    def train_agent(self):
        """
        Main training loop
        """
        best_test_mse = 1e10
        best_return = -1e10
        generator = iter(self.train_loader)

        for step in tqdm(range(self.train_steps), dynamic_ncols=True):
            # evaluate
            if not step % self.eval_every:
                log_info = {
                    "total_mse": [],
                    "first_mse": [],
                    "last_mse": [],
                    "output_divergence": [],
                }
                log_info_means = {}

                for batch in tqdm(
                    self.test_loader, desc="Evaluating", position=0, leave=True
                ):
                    info = self.evaluate(batch)
                    for key in log_info:
                        log_info[key].append(info[key])
                for key in log_info:
                    log_info_means[key] = sum(log_info[key]) / len(log_info[key])
                if log_info_means["total_mse"] < best_test_mse:
                    best_test_mse = log_info["total_mse"]
                    self.store_model_weights()
                    log.info("New best test loss. Stored weights have been updated!")
                log_info["lr"] = self.optimizer.param_groups[0]["lr"]

                wandb.log({k: v for k, v in log_info.items()}, step=step)

            # simulate
            if not step % self.sim_every:
                results = self.env.simulate(self)
                wandb.log(results, step=step)

                # save the best model by reward
                # returns = [v for k, v in results.items() if k.endswith("/return_mean")]
                rewards = [v for k, v in results.items() if k.endswith("/reward_mean")]
                max_return = max(rewards)
                if max_return > best_return:
                    best_return = max_return
                    self.store_model_weights(best_reward=True)
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

        self.store_model_weights()
        log.info("Training done!")

    def train_step(self, batch: dict):
        data_dict = self.process_batch(batch)

        self.model.train()
        self.model.training = True

        action = data_dict["action"]
        noise = torch.randn_like(action)
        if self.use_ddpm:
            timesteps = torch.randint(0, self.sampling_steps, (noise.shape[0],))
            noise_trajectory = self.noise_scheduler.add_noise(action, noise, timesteps)
            timesteps = timesteps.float().to(self.device)
            pred = self.model(noise_trajectory, timesteps, data_dict)
            loss = torch.nn.functional.mse_loss(pred, noise)
        else:
            sigma = self.make_sample_density(len(noise))
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
        if self.use_ddpm:
            self.noise_scheduler.set_timesteps(self.sampling_steps)
            x_0 = self.sample_ddpm(noise, data_dict)

            data_dict["return"] = torch.ones_like(data_dict["return"])
            x_0_max_return = self.sample_ddpm(noise, data_dict)
        else:
            noise = noise * self.sigma_max
            sigmas = utils.get_sigmas_exponential(
                self.sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
            x_0 = self.sample_ddim(noise, sigmas, data_dict, predict=False)

            # data_dict["return"] = torch.ones_like(data_dict["return"])
            x_0_max_return = self.sample_ddim(noise, sigmas, data_dict, predict=False)

        x_0 = self.scaler.inverse_scale_output(x_0)
        x_0_max_return = self.scaler.inverse_scale_output(x_0_max_return)
        data_dict["action"] = self.scaler.inverse_scale_output(data_dict["action"])

        # calculate the MSE
        mse = nn.functional.mse_loss(x_0, data_dict["action"], reduction="none")
        total_mse = mse.mean().item()
        first_mse = mse[:, 0, :].mean().item()
        last_mse = mse[:, -1, :].mean().item()

        output_divergence = torch.abs(x_0 - x_0_max_return).mean().item()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())

        info = {
            "total_mse": total_mse,
            "first_mse": first_mse,
            "last_mse": last_mse,
            "output_divergence": output_divergence,
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
            n_sampling_steps = self.sampling_steps

        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        self.model.eval()

        # get the sigma distribution for the desired sampling method
        noise = torch.randn(
            (self.num_envs, self.T, self.action_dim), device=self.device
        )
        if self.use_ddpm:
            self.noise_scheduler.set_timesteps(n_sampling_steps)
            x_0 = self.sample_ddpm(noise, data_dict, predict=True)
        else:
            noise *= self.sigma_max
            sigmas = utils.get_sigmas_exponential(
                n_sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
            # sigmas = utils.get_sigmas_linear(
            #     n_sampling_steps, self.sigma_min, self.sigma_max, self.device
            # )

            x_0 = self.sample_ddim(noise, sigmas, data_dict, predict=True)
            # x_0 = self.sample_euler_ancestral(noise, sigmas, data_dict, predict=True)
            # x_0 = self.sample_dpmpp_2m_sde(noise, sigmas, data_dict, predict=True)

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

        for i in range(len(sigmas) - 1):
            if predict:
                denoised = self.cfg_forward(x_t, sigmas[i] * s_in, data_dict)
            else:
                denoised = self.model(x_t, sigmas[i] * s_in, data_dict)
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            x_t = ((-t_next).exp() / (-t).exp()) * x_t - (-h).expm1() * denoised

        return x_t

    @torch.no_grad()
    def sample_euler_ancestral(
        self,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        data_dict: dict,
        predict: bool = False,
    ):
        """
        Ancestral sampling with Euler method steps.

        1. compute dx_{i}/dt at the current timestep
        2. get sigma_{up} and sigma_{down} from ancestral method
        3. compute x_{t-1} = x_{t} + dx_{t}/dt * sigma_{down}
        4. Add additional noise after the update step x_{t-1} =x_{t-1} + z * sigma_{up}
        """
        x_t = noise
        s_in = x_t.new_ones([x_t.shape[0]])
        for i in range(len(sigmas) - 1):
            # compute x_{t-1}
            if predict:
                denoised = self.cfg_forward(x_t, sigmas[i] * s_in, data_dict)
            else:
                denoised = self.model(x_t, sigmas[i] * s_in, data_dict)
            # get ancestral steps
            sigma_down, sigma_up = utils.get_ancestral_step(sigmas[i], sigmas[i + 1])
            # compute dx/dt
            d = (x_t - denoised) / sigmas[i]
            # compute dt based on sigma_down value
            dt = sigma_down - sigmas[i]
            # update current action
            x_t = x_t + d * dt
            if sigma_down > 0:
                x_t = x_t + torch.randn_like(x_t) * sigma_up

        return x_t

    @torch.no_grad()
    def sample_dpmpp_2m_sde(
        self,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        data_dict: dict,
        predict: bool = False,
    ):
        """DPM-Solver++(2M)."""
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        x_t = noise
        noise_sampler = utils.BrownianTreeNoiseSampler(x_t, sigma_min, sigma_max)
        s_in = x_t.new_ones([x_t.shape[0]])

        old_denoised = None
        h_last = None

        for i in range(len(sigmas) - 1):
            if predict:
                denoised = self.cfg_forward(x_t, sigmas[i] * s_in, data_dict)
            else:
                denoised = self.model(x_t, sigmas[i] * s_in, data_dict)

            # DPM-Solver++(2M) SDE
            if sigmas[i + 1] == 0:
                x_t = denoised
            else:
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = h

                x_t = (
                    sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x_t
                    + (-h - eta_h).expm1().neg() * denoised
                )

                if old_denoised is not None:
                    r = h_last / h
                    x_t = x_t + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (
                        1 / r
                    ) * (denoised - old_denoised)

                x_t = (
                    x_t
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                )

            old_denoised = denoised
            h_last = h
        return x_t

    @torch.no_grad()
    def sample_ddpm(self, noise: torch.Tensor, data_dict: dict, predict: bool = False):
        """
        Perform inference using the DDPM sampler
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

    def load_pretrained_model(self, weights_path: str) -> None:
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
        self.scaler.x_mean = scaler_state["x_mean"]
        self.scaler.x_std = scaler_state["x_std"]
        self.scaler.y_mean = scaler_state["y_mean"]
        self.scaler.y_std = scaler_state["y_std"]

        log.info("Loaded pre-trained model parameters and scaler")

    def store_model_weights(self, best_reward: bool = False) -> None:
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        name = (
            "model_state_dict.pth" if not best_reward else "best_model_state_dict.pth"
        )
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, name))
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(
            self.model.state_dict(),
            os.path.join(self.output_dir, "non_ema_" + name),
        )

        # Save scaler attributes
        torch.save(
            {
                "x_max": self.scaler.x_max,
                "x_min": self.scaler.x_min,
                "y_max": self.scaler.y_max,
                "y_min": self.scaler.y_min,
                "x_mean": self.scaler.x_mean,
                "x_std": self.scaler.x_std,
                "y_mean": self.scaler.y_mean,
                "y_std": self.scaler.y_std,
            },
            os.path.join(self.output_dir, "scaler.pth"),
        )

    @torch.no_grad()
    def make_sample_density(self, size):
        """
        Generate a density function for training sigmas
        """
        loc = math.log(self.sigma_data)
        density = utils.rand_log_logistic(
            (size,), loc, 0.5, self.sigma_min, self.sigma_max, self.device
        )
        return density

    @torch.no_grad()
    def process_batch(self, batch: dict) -> dict:
        batch = self.dict_to_device(batch)

        raw_obs = batch["obs"]
        raw_action = batch.get("action", None)
        skill = batch["skill"]
        vel_cmd = batch.get("vel_cmd", None)
        # if vel_cmd is None:
        #     vel_cmd = self.sample_vel_cmd(raw_obs.shape[0])

        returns = batch.get("return", None)
        if returns is None:
            returns = self.compute_returns(raw_obs, vel_cmd)

        rewards = batch.get("reward", None)
        if rewards is not None and returns is None:
            returns = self.compute_returns_from_rewards(rewards)

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
        # vel_ranges = torch.tensor([0.8, 0.5, 1.0], device=self.device)
        # vel_cmd = torch.rand((batch_size, 3), device=self.device)
        # vel_cmd = vel_cmd * 2 * vel_ranges - vel_ranges
        # return vel_cmd
        vel_cmd = torch.randint(0, 2, (batch_size, 1), device=self.device).float()
        return vel_cmd * 2 - 1

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

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 2)
        # axs[0].hist(rewards.flatten().cpu().numpy(), bins=20)
        # axs[0].set_xlabel("Rewards")
        # axs[0].set_ylabel("Frequency")
        # axs[1].hist(returns.flatten().cpu().numpy(), bins=20)
        # axs[1].set_xlabel("Returns")
        # axs[1].set_ylabel("Frequency")
        # plt.savefig("returns.png")
        # print(returns.max())
        # exit()

        return returns.unsqueeze(-1)

    def compute_returns_from_rewards(self, rewards):
        rewards = rewards.clone() - 0.22  # max reward
        horizon = 50
        gammas = torch.tensor([0.99**i for i in range(horizon)]).to(self.device)
        returns = (rewards * gammas).sum(dim=-1)
        returns = torch.exp(returns / 50)
        returns = (returns - returns.min()) / (returns.max() - returns.min())

        return returns.unsqueeze(-1)
