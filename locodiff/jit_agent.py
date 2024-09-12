import logging
import math
import os

import hydra
import torch

import locodiff.utils as utils

log = logging.getLogger(__name__)


class JitAgent:

    def __init__(
        self,
        model,
        dataset_fn,
        device: str,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        T: int,
        T_cond: int,
        T_action: int,
        obs_dim: int,
        action_dim: int,
        cond_lambda: int,
    ):
        # model
        self.model = hydra.utils.instantiate(model).to(device).eval()
        self.model.inner_model.detach_all()
        self.T = T
        self.T_cond = T_cond
        self.T_action = T_action
        self.obs_hist = torch.zeros((1, T_cond, obs_dim), device=device)

        # diffusion
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cond_lambda = cond_lambda

        # env
        self.action_dim = action_dim

        self.scaler = hydra.utils.instantiate(dataset_fn)[-1]
        self.device = device

    @torch.no_grad()
    def forward(self, obs, vel_cmd, skill, returns) -> torch.Tensor:
        """
        Predicts the output of the model based on the provided batch of data.
        """
        use_cfg = False

        batch = {"obs": obs, "vel_cmd": vel_cmd, "skill": skill, "return": returns}
        data_dict = self.process_batch(batch)

        # get the sigma distribution for the desired sampling method
        noise = torch.randn((1, self.T, self.action_dim), device=self.device)
        noise *= self.sigma_max
        sigmas = utils.get_sigmas_exponential(
            self.num_sampling_steps, self.sigma_min, self.sigma_max, self.device
        )
        x_0 = self.sample_ddim(noise, sigmas, data_dict, use_cfg)

        # get the action for the current timestep
        x_0 = self.scaler.clip(x_0)
        pred_action = self.scaler.inverse_scale_output(x_0)
        pred_action = pred_action[:, : self.T_action]

        return pred_action

    def stack_context(self, batch):
        self.obs_hist[:, :-1] = self.obs_hist[:, 1:].clone()
        self.obs_hist[:, -1] = batch["obs"]
        batch["obs"] = self.obs_hist.clone()
        return batch

    @torch.no_grad()
    def sample_ddim(
        self, noise: torch.Tensor, sigmas: torch.Tensor, data_dict: dict, use_cfg: bool
    ):
        """
        Perform inference using the DDIM sampler
        """
        x_t = noise
        s_in = x_t.new_ones([x_t.shape[0]])

        for i in range(sigmas.shape[0] - 1):
            if use_cfg:
                denoised = self.cfg_forward(x_t, sigmas[i] * s_in, data_dict)
            else:
                denoised = self.model(x_t, sigmas[i] * s_in, data_dict)
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            x_t = ((-t_next).exp() / (-t).exp()) * x_t - (-h).expm1() * denoised

        return x_t

    def cfg_forward(self, x_t: torch.Tensor, sigma: torch.Tensor, data_dict: dict):
        """
        Classifier-free guidance sample
        """
        # TODO: parallelize this

        out = self.model(x_t, sigma, data_dict)

        out_uncond = self.model(x_t, sigma, data_dict, uncond=True)
        out = out_uncond + self.cond_lambda * (out - out_uncond)

    def load_pretrained_model(self, weights_path: str) -> None:
        self.model.load_state_dict(
            torch.load(
                os.path.join(weights_path, "model_state_dict.pth"),
                map_location=self.device,
            ),
            strict=False,
        )

        # Load scaler attributes
        scaler_state = torch.load(
            os.path.join(weights_path, "scaler.pth"), map_location=self.device
        )
        self.scaler.x_max = scaler_state["x_max"]
        self.scaler.x_min = scaler_state["x_min"]
        self.scaler.y_max = scaler_state["y_max"]
        self.scaler.y_min = scaler_state["y_min"]
        self.scaler.x_mean = scaler_state["x_mean"]
        self.scaler.x_std = scaler_state["x_std"]
        self.scaler.y_mean = scaler_state["y_mean"]
        self.scaler.y_std = scaler_state["y_std"]

        log.info("Loaded pre-trained model parameters and scaler")

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
    def process_batch(self, batch: dict) -> dict:
        batch = self.dict_to_device(batch)

        raw_obs = batch["obs"]
        raw_action = batch.get("action", None)
        skill = batch["skill"]
        vel_cmd = batch.get("vel_cmd", None)
        returns = batch.get("return", None)
        obs = self.scaler.scale_input(raw_obs[:, : self.T_cond])

        if raw_action is None:
            action = None
        else:
            action = self.scaler.scale_output(
                raw_action[:, self.T_cond - 1 : self.T_cond + self.T - 1],
            )

        processed_batch = {
            "obs": obs,
            "action": action,
            "vel_cmd": vel_cmd,
            "skill": skill,
            "return": returns,
        }

        return processed_batch

    def dict_to_device(self, batch):
        return {k: v.clone().to(self.device) for k, v in batch.items()}
