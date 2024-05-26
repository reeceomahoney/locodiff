import torch
import hydra
import math
import os
from faulthandler import disable
from functools import partial
import logging

import hydra
import torch
from tqdm import trange

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
        pred_obs_dim: int,
        action_dim: int,
    ):
        # model
        self.model = hydra.utils.instantiate(model).to(device).eval()
        self.model.inner_model.detach_all()
        self.T = T  # set to 0 to just predict the next action
        self.T_cond = T_cond
        self.T_action = T_action

        # diffusion
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # env
        self.pred_obs_dim = pred_obs_dim
        self.action_dim = action_dim

        self.scaler = hydra.utils.instantiate(dataset_fn)[-1]
        self.device = device

    @torch.no_grad()
    def forward(self, obs, goal) -> torch.Tensor:
        """
        Predicts the output of the model based on the provided batch of data.
        """
        state_in, goal = self.process_batch({"observation": obs, "goal": goal})

        # get the sigma distribution for the desired sampling method
        sigmas = utils.get_sigmas_exponential(
            self.num_sampling_steps, self.sigma_min, self.sigma_max, self.device
        )

        sa_dim = self.pred_obs_dim + self.action_dim
        x = torch.ones((1, self.T, sa_dim), device=self.device)
        x *= self.sigma_max

        x_0 = self.sample_ddim(x, sigmas, state_in, goal)

        # get the action for the current timestep
        x_0 = self.scaler.clip(x_0)
        pred_traj = self.scaler.inverse_scale_output(x_0)
        pred_action = pred_traj[:, : self.T_action, self.pred_obs_dim :]

        return pred_action

    @torch.no_grad()
    def sample_ddim(self, x_t, sigmas, cond, goal):
        """
        Sample from the model using the DDIM sampler

        Args:
            x_t (torch.Tensor): The initial state-action noise tensor.
            goals (torch.Tensor): One-hot encoding of the goals.
            sigmas (torch.Tensor): The sigma distribution for the sampling.
            cond (torch.Tensor): The conditioning input for the model.
        Returns:
            torch.Tensor: The predicted output of the model.
        """
        s_in = x_t.new_ones([x_t.shape[0]])

        for i in trange(sigmas.shape[0] - 1, disable=disable):
            denoised = self.model(x_t, cond, sigmas[i] * s_in, goal=goal)
            t, t_next = self.t_fn(sigmas[i]), self.t_fn(sigmas[i + 1])
            h = t_next - t
            x_t = (self.sigma_fn(t_next) / self.sigma_fn(t)) * x_t - (
                -h
            ).expm1() * denoised

        return x_t

    def sigma_fn(self, t):
        return t.neg().exp()

    def t_fn(self, sigma):
        return sigma.log().neg()

    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
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
        self.scaler.goal_max = scaler_state["goal_max"]
        self.scaler.goal_min = scaler_state["goal_min"]

        log.info("Loaded pre-trained model parameters and scaler")

    @torch.no_grad()
    def make_sample_density(self):
        """
        Generate a density function for training sigmas
        """
        sd_config = []
        loc = sd_config["loc"] if "loc" in sd_config else math.log(self.sigma_data)
        scale = sd_config["scale"] if "scale" in sd_config else 0.5
        min_value = (
            sd_config["min_value"] if "min_value" in sd_config else self.sigma_min
        )
        max_value = (
            sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
        )
        return partial(
            utils.rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )

    @torch.no_grad()
    def process_batch(self, batch: dict):
        """
        Processes a batch of data and returns the state, action and goal
        """
        state = self.get_to_device(batch, "observation")
        goal = self.get_to_device(batch, "goal")

        # Centre posisition around the current state
        current_pos = state[:, self.T_cond - 1, :2].clone()
        state[..., :2] = state[..., :2] - current_pos.unsqueeze(1)
        state_in = self.scaler.scale_input(state[:, : self.T_cond])

        goal[..., :2] -= current_pos
        goal = self.scaler.scale_goal(goal)

        return state_in, goal

    def get_to_device(self, batch, key):
        return batch.get(key).clone().to(self.device)
