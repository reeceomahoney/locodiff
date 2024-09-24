import torch
import torch.nn as nn


class CFGWrapper(nn.Module):
    """
    Classifier-free guidance wrapper
    """

    def __init__(self, model, cond_lambda: int, cond_mask_prob: float):
        super().__init__()
        self.model = model
        self.cond_lambda = cond_lambda
        self.cond_mask_prob = cond_mask_prob

    def __call__(self, x_t: torch.Tensor, sigma: torch.Tensor, data_dict: dict):
        out = self.model(x_t, sigma, data_dict)
        out_uncond = self.model(x_t, sigma, data_dict, uncond=True)
        out = out_uncond + self.cond_lambda * (out - out_uncond)

        return out

    def loss(self, noise, sigma, data_dict):
        return self.model.loss(noise, sigma, data_dict)

    def get_params(self):
        return self.model.get_params()

    def get_optim_groups(self):
        return self.model.get_optim_groups()


class ScalingWrapper(nn.Module):
    """
    Wrapper for diffusion transformer that applies scaling from Karras et al. 2022
    """

    def __init__(self, sigma_data: float, model):
        super().__init__()
        self.sigma_data = sigma_data
        self.model = model

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5

        return c_skip.view(-1, 1, 1), c_out.view(-1, 1, 1), c_in.view(-1, 1, 1)

    def loss(self, noise, sigma, data_dict):
        action = data_dict["action"]

        noised_action = action + noise * sigma.view(-1, 1, 1)

        c_skip, c_out, c_in = self.get_scalings(sigma)
        model_output = self.model(noised_action * c_in, sigma, data_dict)
        target = (action - c_skip * noised_action) / c_out

        loss = (model_output - target).pow(2).mean()
        return loss

    def forward(self, x_t, sigma, data_dict, uncond=False):
        c_skip, c_out, c_in = self.get_scalings(sigma)
        return self.model(x_t * c_in, sigma, data_dict, uncond) * c_out + x_t * c_skip

    def get_params(self):
        return self.model.get_params()

    def get_optim_groups(self):
        optim_groups = self.model.get_optim_groups()
        return optim_groups
