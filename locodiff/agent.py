import math

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from locodiff.samplers import get_sampler, get_sigmas_exponential, rand_log_logistic
from locodiff.wrappers import CFGWrapper


class Agent(nn.Module):
    def __init__(
        self,
        model,
        noise_scheduler: DDPMScheduler,
        obs_dim: int,
        action_dim: int,
        T: int,
        T_cond: int,
        num_envs: int,
        sampling_steps: int,
        sampler_type: str,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        cond_lambda: int,
        cond_mask_prob: float,
        device,
    ):
        super().__init__()

        # model
        if cond_mask_prob > 0:
            model = CFGWrapper(model, cond_lambda, cond_mask_prob)
        self.model = model
        self.sampler = get_sampler(sampler_type)

        self.noise_scheduler = noise_scheduler
        self.device = device

        # dims
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.T = T
        self.T_cond = T_cond
        self.num_envs = num_envs

        # diffusion
        self.sampling_steps = sampling_steps
        self.sampler_type = sampler_type
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_mask_prob = cond_mask_prob

        # ddpm
        self.noise_scheduler = noise_scheduler

    def __call__(self, data_dict: dict, **kwargs) -> tuple:
        self.eval()

        if data_dict["action"] is None:
            batch_size = self.num_envs
        else:
            batch_size = data_dict["action"].shape[0]

        noise = torch.randn((batch_size, self.T, self.action_dim)).to(
            self.device
        )
        if self.sampler_type == "ddpm":
            self.noise_scheduler.set_timesteps(self.sampling_steps)
            kwargs = {"noise_scheduler": self.noise_scheduler}
        else:
            noise = noise * self.sigma_max
            sigmas = get_sigmas_exponential(
                self.sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
            kwargs = {"sigmas": sigmas}
        x_0 = self.sampler(self.model, noise, data_dict, **kwargs)

        if self.cond_mask_prob > 0:
            data_dict["return"] = torch.ones_like(data_dict["return"])
            x_0_max_return = self.sampler(self.model, noise, data_dict, **kwargs)
            return x_0, x_0_max_return
        else:
            return x_0

    def loss(self, data_dict) -> torch.Tensor:
        self.train()

        action = data_dict["action"]
        noise = torch.randn_like(action)
        if self.sampler_type == "ddpm":
            timesteps = torch.randint(0, self.sampling_steps, (noise.shape[0],))
            noise_trajectory = self.noise_scheduler.add_noise(action, noise, timesteps)
            timesteps = timesteps.float().to(self.device)
            pred = self.model(noise_trajectory, timesteps, data_dict)
            loss = torch.nn.functional.mse_loss(pred, noise)
        else:
            sigma = self.make_sample_density(len(noise))
            loss = self.model.loss(noise, sigma, data_dict)

        return loss

    @torch.no_grad()
    def make_sample_density(self, size):
        """
        Generate a density function for training sigmas
        """
        loc = math.log(self.sigma_data)
        density = rand_log_logistic(
            (size,), loc, 0.5, self.sigma_min, self.sigma_max, self.device
        )
        return density

    def get_params(self):
        return self.model.get_params()

    def get_optim_groups(self):
        return self.model.get_optim_groups()
