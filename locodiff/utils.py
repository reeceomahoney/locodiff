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

    def update_pos_scale(self, batch, T_cond):
        obs_batch = batch["observation"]
        goal_batch = batch["goal"]

        pos = obs_batch[..., :2] - obs_batch[:, T_cond - 1, :2].unsqueeze(1)
        pos_flat_in = pos[:, :T_cond].reshape(-1, 2)
        self.x_max[:2] = pos_flat_in.max(dim=0).values.to(self.device)
        self.x_min[:2] = pos_flat_in.min(dim=0).values.to(self.device)

        pos_flat_out = pos[:, T_cond - 1 :].reshape(-1, 2)
        self.y_max[:2] = pos_flat_out.max(dim=0).values.to(self.device)
        self.y_min[:2] = pos_flat_out.min(dim=0).values.to(self.device)

        goal = goal_batch[..., :2] - obs_batch[:, T_cond - 1, :2]
        self.goal_min = goal.min(dim=0).values.to(self.device)
        self.goal_max = goal.max(dim=0).values.to(self.device)

    def scale_input(self, x):
        out = (x - self.x_min) / (self.x_max - self.x_min) * 2 - 1
        return out

    def scale_output(self, y):
        out = (y - self.y_min) / (self.y_max - self.y_min) * 2 - 1
        return out

    def inverse_scale_output(self, y):
        out = (y + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        return out
    
    def scale_goal(self, goal):
        return (goal - self.goal_min) / (self.goal_max - self.goal_min) * 2 - 1

    def clip(self, y):
        return torch.clamp(y, self.y_bounds[0, :], self.y_bounds[1, :])
