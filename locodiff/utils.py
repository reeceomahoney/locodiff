import math
import numpy as np
import torch
import torch.nn as nn


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
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


def reward_function(obs, vel_cmds, fn_name):
    x_vel = obs[..., 30]
    z_vel = obs[..., 17]
    if x_vel.ndim == 1:
        vel_cmds = vel_cmds.squeeze(1)

    if fn_name == "x_vel":
        rewards = x_vel
        rewards = torch.clamp(rewards, -0.6, 0.6)
    elif fn_name == "fwd_bwd":
        rewards = torch.zeros_like(x_vel)
        rewards = torch.where(vel_cmds == 1, x_vel, rewards)
        rewards = torch.where(vel_cmds == 0, -x_vel, rewards)
        rewards = torch.clamp(rewards, -0.6, 0.6)
    elif fn_name == "on_off":
        rewards = torch.where(vel_cmds == 1, x_vel, 1)
    elif fn_name == "vel_target":
        lin_vel = obs[..., 30:32]
        ang_vel = obs[..., 17:18]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)
        vel_cmds = torch.tensor([0.8, 0.0, 0.0]).to(vel.device)
        rewards = torch.exp(-3*((vel - vel_cmds) ** 2)).mean(dim=-1)
    elif fn_name == "self_cmd":
        rewards = torch.zeros_like(x_vel)
        rewards = torch.where(z_vel >= 0, x_vel, rewards)
        rewards = torch.where(z_vel < 0, -x_vel, rewards)
        rewards = torch.clamp(rewards, -0.6, 0.6)
    else:
        raise ValueError(f"Unknown reward function {fn_name}")

    return rewards


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# copied from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, device: str = "cuda", use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self._device = device
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

        self.steps = 0

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_shadow_params(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.data.copy_(param.data)

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]


class MinMaxScaler:
    """
    Min Max scaler, that scales the output data between -1 and 1 and the input to a uniform Gaussian.
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, device: str):
        self.device = device

        x_data = x_data.detach()
        y_data = y_data.detach()

        self.x_max = x_data.max(0).values.to(device)
        self.x_min = x_data.min(0).values.to(device)

        self.y_max = y_data.max(0).values.to(device)
        self.y_min = y_data.min(0).values.to(device)

        self.y_bounds = torch.zeros((2, y_data.shape[-1])).to(device)
        self.y_bounds[0, :] = -1.1
        self.y_bounds[1, :] = 1.1

    def scale_input(self, x):
        out = (x - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        return out

    def scale_output(self, y):
        out = (y - self.y_min) / (self.y_max - self.y_min) * 2 - 1
        return out

    def inverse_scale_output(self, y):
        out = (y + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        return out

    def clip(self, y):
        return torch.clamp(y, self.y_bounds[0, :], self.y_bounds[1, :])
