from typing import Callable

import torch

import locodiff.utils as utils


def get_sampler(sampler_type: str) -> Callable:
    if sampler_type == "ddim":
        return sample_ddim
    elif sampler_type == "dpmpp_2m_sde":
        return sample_dpmpp_2m_sde
    elif sampler_type == "euler_ancestral":
        return sample_euler_ancestral
    elif sampler_type == "ddpm":
        return sample_ddpm
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


@torch.no_grad()
def sample_ddim(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """
    Perform inference using the DDIM sampler
    """
    sigmas = kwargs["sigmas"]
    x_t = noise
    s_in = x_t.new_ones([x_t.shape[0]])

    for i in range(len(sigmas) - 1):
        denoised = model(x_t, sigmas[i] * s_in, data_dict)
        t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
        h = t_next - t
        x_t = ((-t_next).exp() / (-t).exp()) * x_t - (-h).expm1() * denoised

    return x_t


@torch.no_grad()
def sample_euler_ancestral(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """
    Ancestral sampling with Euler method steps.

    1. compute dx_{i}/dt at the current timestep
    2. get sigma_{up} and sigma_{down} from ancestral method
    3. compute x_{t-1} = x_{t} + dx_{t}/dt * sigma_{down}
    4. Add additional noise after the update step x_{t-1} =x_{t-1} + z * sigma_{up}
    """
    sigmas = kwargs["sigmas"]
    x_t = noise
    s_in = x_t.new_ones([x_t.shape[0]])
    for i in range(len(sigmas) - 1):
        # compute x_{t-1}
        denoised = model(x_t, sigmas[i] * s_in, data_dict)
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
def sample_dpmpp_2m_sde(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """DPM-Solver++(2M)."""
    sigmas = kwargs["sigmas"]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    x_t = noise
    noise_sampler = utils.BrownianTreeNoiseSampler(x_t, sigma_min, sigma_max)
    s_in = x_t.new_ones([x_t.shape[0]])

    old_denoised = None
    h_last = None

    for i in range(len(sigmas) - 1):
        denoised = model(x_t, sigmas[i] * s_in, data_dict)

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
def sample_ddpm(model, noise: torch.Tensor, data_dict: dict, **kwargs):
    """
    Perform inference using the DDPM sampler
    """
    noise_scheduler = kwargs["noise_scheduler"]
    x_t = noise

    for t in noise_scheduler.timesteps:
        t_pt = t.float().to(noise.device)
        output = model(x_t, t_pt.expand(x_t.shape[0]), data_dict)
        x_t = noise_scheduler.step(output, t, x_t).prev_sample

    return x_t
