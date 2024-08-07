import hydra
import torch


class ScalingWrapper(torch.nn.Module):
    """
    Wrapper for diffusion transformer that applies scaling from Karras et al. 2022
    """

    def __init__(self, inner_model, sigma_data):
        super().__init__()
        self.inner_model = hydra.utils.instantiate(inner_model)
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5

        return c_skip.view(-1, 1, 1), c_out.view(-1, 1, 1), c_in.view(-1, 1, 1)

    def loss(self, noise, sigma, data_dict):
        action = data_dict["action"]

        noised_action = action + noise * sigma.view(-1, 1, 1)

        c_skip, c_out, c_in = self.get_scalings(sigma)
        model_output = self.inner_model(noised_action * c_in, sigma, data_dict)
        target = (action - c_skip * noised_action) / c_out

        loss = (model_output - target).pow(2).mean()
        return loss

    def forward(self, x_t, sigma, data_dict, uncond=False):
        c_skip, c_out, c_in = self.get_scalings(sigma)
        return (
            self.inner_model(x_t * c_in, sigma, data_dict, uncond) * c_out + x_t * c_skip
        )

    def get_params(self):
        return self.inner_model.parameters()
