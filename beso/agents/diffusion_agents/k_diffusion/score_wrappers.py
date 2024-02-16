from multiprocessing.sharedctypes import Value

import hydra
from torch import DictType, nn
from .utils import append_dims
import torch

"""
Wrappers for the score-based models based on Karras et al. 2022
They are used to get improved scaling of different noise levels, which
improves training stability and model performance 

Code is adapted from:

https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
"""


class GCDenoiser(nn.Module):
    """
    A Karras et al. preconditioner for denoising diffusion models.

    Args:
        inner_model: The inner model used for denoising.
        sigma_data: The data sigma for scalings (default: 1.0).
    """

    def __init__(self, inner_model, sigma_data, T_cond):
        super().__init__()
        self.inner_model = hydra.utils.instantiate(inner_model)
        self.sigma_data = sigma_data
        self.T_cond = T_cond
        self.obs_dim = inner_model.obs_dim

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state_action, goal, noise, sigma, **kwargs):
        """
        Compute the loss for the denoising process.

        Args:
            state_action: The input state_action.
            goal: The input goal.
            noise: The input noise.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.
        Returns:
            The computed loss.
        """
        # split into past and future states
        # last action in history and first observation in future are set to 0
        cond = state_action[:, : self.T_cond, :]
        cond[:, -1, self.obs_dim :] = 0
        sa_x = state_action[:, self.T_cond :, :]
        sa_x[:, 0, : self.obs_dim] = 0

        noised_input = sa_x + noise * append_dims(sigma, state_action.ndim)

        c_skip, c_out, c_in = [
            append_dims(x, state_action.ndim) for x in self.get_scalings(sigma)
        ]
        model_output = self.inner_model(noised_input * c_in, cond, sigma, **kwargs)
        target = (sa_x - c_skip * noised_input) / c_out

        # remove the first obs from the loss
        loss = (model_output - target).pow(2).flatten(1)
        return loss[:, self.obs_dim :].mean()

    def forward(self, x_t, cond, sigma, **kwargs):
        """
        Perform the forward pass of the denoising process.

        Args:
            state_action: The input state_action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings(sigma)
        ]
        return (
            self.inner_model(x_t * c_in, cond, sigma, **kwargs) * c_out + x_t * c_skip
        )

    def get_params(self):
        return self.inner_model.parameters()
