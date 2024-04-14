import math
import numpy as np
import torch
import torch.nn as nn


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


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
    Min Max scaler, that scales the output data between -1 and 1 and the input data between 0 and 1.
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, device: str):
        self.device = device

        x_data = x_data.detach().cpu().numpy()
        y_data = y_data.detach().cpu().numpy()

        with torch.no_grad():
            self.y_min = torch.from_numpy(y_data.min(0)).to(device)
            self.y_max = torch.from_numpy(y_data.max(0)).to(device)

            self.new_max_y = torch.ones_like(self.y_max)
            self.new_min_y = -1 * torch.ones_like(self.y_max)

            self.x_mean = torch.from_numpy(x_data.mean(0)).to(device)
            self.x_std = torch.from_numpy(x_data.std(0)).to(device)

            self.y_bounds = np.zeros((2, y_data.shape[-1]))
            self.y_bounds[0, :] = -1 * np.ones_like(y_data.min(0))[:]
            self.y_bounds[1, :] = np.ones_like(y_data.min(0))[:]
            self.y_bounds_tensor = torch.from_numpy(self.y_bounds).to(device)

    @torch.no_grad()
    def scale_input(self, x):
        """
        Scales the input tensor `x` based on the defined scaling parameters.

        Args:
            x (torch.Tensor): The input tensor to be scaled.
        Returns:
            torch.Tensor: The scaled input tensor.
        """
        x = x.to(self.device)
        out = (x - self.x_mean) / (
            self.x_std + 1e-12 * torch.ones((self.x_std.shape), device=self.device)
        )
        return out.to(torch.float32)

    @torch.no_grad()
    def scale_output(self, y):
        """
        Scales the output tensor `y` based on the defined scaling parameters.
        Args:
            y (torch.Tensor): The output tensor to be scaled.
        Returns:
            torch.Tensor: The scaled output tensor.
        """
        y = y.to(self.device)
        out = (y - self.y_min) / (self.y_max - self.y_min) * (
            self.new_max_y - self.new_min_y
        ) + self.new_min_y
        return out.to(torch.float32)

    @torch.no_grad()
    def inverse_scale_input(self, x):
        """
        Inversely scales the input tensor `x` based on the defined scaling parameters.

        Args:
            x (torch.Tensor): The input tensor to be inversely scaled.
        Returns:
            torch.Tensor: The inversely scaled input tensor.
        """
        out = x * (
            self.x_std + 1e-12 * torch.ones((self.x_std.shape), device=self.device)
        )
        out += self.x_mean
        return out.to(torch.float32)

    @torch.no_grad()
    def inverse_scale_output(self, y):
        """
        Inversely scales the output tensor `y` based on the defined scaling parameters.

        Args:
            y (torch.Tensor): The output tensor to be inversely scaled.
        Returns:
            torch.Tensor: The inversely scaled output tensor.
        """
        y.to(self.device)
        out = (y - self.new_min_y) / (self.new_max_y - self.new_min_y) * (
            self.y_max - self.y_min
        ) + self.y_min
        return out

    @torch.no_grad()
    def clip_action(self, y):
        """
        Clips the input tensor `y` based on the defined action bounds.
        """
        return (
            torch.clamp(
                y, self.y_bounds_tensor[0, :] * 1.1, self.y_bounds_tensor[1, :] * 1.1
            )
            .to(self.device)
            .to(torch.float32)
        )


# code adapted from
# https://github.com/GuyTevet/motion-diffusion-model/blob/55cd84e14ba8e0e765700913023259b244f5e4ed/model/cfg_sampler.py
class ClassifierFreeSampleModel(nn.Module):
    """
    A wrapper model that adds conditional sampling capabilities to an existing model.

    Args:
        model (nn.Module): The underlying model to run.
        cond_lambda (float): Optional. The conditional lambda value. Defaults to 2.

    Attributes:
        model (nn.Module): The underlying model.
        cond_lambda (float): The conditional lambda value.
    """

    def __init__(self, model, cond_lambda: float):
        super().__init__()
        # TODO: refactor this into beso agent
        self.model = model
        self.cond_lambda = cond_lambda

    def forward(self, x_t, cond, sigma, goal):

        # unconditional output
        uncond_dict = {"uncond": True}
        out_uncond = self.model(x_t, cond, sigma, goal, **uncond_dict)

        if goal.sum() != 0:
            out_cond = self.model(x_t, cond, sigma, goal)
            out_uncond += self.cond_lambda * (out_cond - out_uncond)

        return out_uncond

    def get_params(self):
        return self.model.get_params()
