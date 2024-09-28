import math
from typing import Callable

import torch
import torchsde


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [
            torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed
        ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.
    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(
            torch.as_tensor(sigma_max)
        )
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(
            torch.as_tensor(sigma_next)
        )
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


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

    num_steps = kwargs.get("num_steps", len(sigmas) - 1)

    for i in range(num_steps):
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
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
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
    noise_sampler = BrownianTreeNoiseSampler(x_t, sigma_min, sigma_max)
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


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_linear(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an linear noise schedule."""
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def rand_log_logistic(
    shape,
    loc=0.0,
    scale=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)
