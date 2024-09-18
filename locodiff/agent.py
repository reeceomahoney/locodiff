import math

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import locodiff.utils as utils


class Agent(nn.Module):
    def __init__(
        self,
        model,
        noise_scheduler: DDPMScheduler,
        action_dim: int,
        T: int,
        T_cond: int,
        num_envs: int,
        sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        cond_lambda: int,
        cond_mask_prob: float,
        device,
        use_ddpm: bool,
    ):
        super().__init__()

        # model
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device

        # dims
        self.action_dim = action_dim
        self.T = T
        self.T_cond = T_cond
        self.num_envs = num_envs

        # diffusion
        self.sampling_steps = sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_lambda = cond_lambda
        self.cond_mask_prob = cond_mask_prob

        # ddpm
        self.use_ddpm = use_ddpm
        self.noise_scheduler = noise_scheduler

    def __call__(self, data_dict: dict) -> tuple:
        self.eval()
        self.training = False

        if data_dict["action"] is None:
            batch_size = self.num_envs
        else:
            batch_size = data_dict["action"].shape[0]

        noise = torch.randn((batch_size, self.T, self.action_dim)).to(self.device)
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
            # sigmas = utils.get_sigmas_linear(
            #     n_sampling_steps, self.sigma_min, self.sigma_max, self.device
            # )

            x_0 = self.sample_ddim(noise, sigmas, data_dict, predict=True)
            # x_0 = self.sample_euler_ancestral(noise, sigmas, data_dict, predict=True)
            # x_0 = self.sample_dpmpp_2m_sde(noise, sigmas, data_dict, predict=True)

            data_dict["return"] = torch.ones_like(data_dict["return"])
            x_0_max_return = self.sample_ddim(noise, sigmas, data_dict, predict=False)

        return x_0, x_0_max_return

    def loss(self, data_dict) -> torch.Tensor:
        self.train()
        self.training = True

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

        return loss

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

    def get_params(self):
        return self.model.get_params()

    def get_optim_groups(self):
        return self.model.get_optim_groups()
