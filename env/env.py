import logging
import os
import platform
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from env.lib.raisim_env import RaisimWrapper
from locodiff.utils import reward_function

log = logging.getLogger(__name__)


class RaisimEnv:

    def __init__(
        self,
        impl: DictConfig,
        seed: int,
        T: int,
        T_cond: int,
        T_action: int,
        skill_dim: int,
        eval_times: int,
        eval_steps: int,
        reward_fn: str,
        device: str,
        lambda_values: list,
        cond_mask_prob: float,
    ):
        if platform.system() == "Darwin":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        self.device = device

        # initialize environment
        resource_dir = os.path.dirname(os.path.realpath(__file__)) + "/resources"
        impl_cfg = OmegaConf.to_yaml(impl)
        self.env = RaisimWrapper(resource_dir, impl_cfg)
        self.env.setSeed(seed)
        self.env.turnOnVisualization()

        # env dims
        self.num_obs = self.env.getObDim()
        self.num_acts = self.env.getActionDim()
        self.num_envs = self.env.getNumOfEnvs()

        # model dims
        self.T_action = T_action
        self.window = T_cond + T - 1
        self.skill_dim = skill_dim

        # initialize containers
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._base_position = np.zeros([self.num_envs, 3], dtype=np.float32)
        self._base_orientation = np.zeros([self.num_envs, 4], dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._torques = np.zeros([self.num_envs, 12], dtype=np.float32)
        self.nominal_joint_pos = np.zeros([self.num_envs, 12], dtype=np.float32)
        self.env.getNominalJointPositions(self.nominal_joint_pos)

        # inference
        self.lambda_values = lambda_values
        self.skill = torch.zeros(self.num_envs, self.skill_dim).to(self.device)
        self.set_skill(0)
        self.vel_cmd = (
            torch.randint(0, 2, (self.num_envs, 1), device=self.device).float() * 2 - 1
        )
        self.target_returns = torch.ones((self.num_envs, 1)).to(self.device)

        # misc
        self.eval_times = eval_times
        self.eval_steps = eval_steps
        self.reward_fn = reward_fn
        self.cond_mask_prob = cond_mask_prob

    ############
    # Main API #
    ############

    def simulate(self, ws, real_time=False, lambda_values=[]):
        if self.cond_mask_prob > 0:
            lambda_tensor, lambda_values = self.set_lambdas(lambda_values)
            prev_lambda = ws.agent.model.cond_lambda
            ws.agent.model.cond_lambda = lambda_tensor

        total_rewards, total_height_rewards, total_dones = [], [], []
        return_dict = {}

        for _ in range(self.eval_times):
            self.env.reset()
            ws.reset(np.ones(self.num_envs, dtype=bool))

            obs, vel_cmd = self.observe()
            action = self.nominal_joint_pos
            done = np.array([False])

            t = 0
            with trange(self.eval_steps, desc="Simulating") as pbar:
                while t < self.eval_steps:
                    start = time.time()
                    if t == 125:
                        self.set_skill(1)

                    pred_action = ws(
                        {
                            "obs": obs,
                            "skill": self.skill,
                            "vel_cmd": vel_cmd,
                            "return": self.target_returns,
                        },
                    )

                    for i in range(self.T_action):
                        # Apply action with delay
                        obs, vel_cmd, rewards, done = self.step(action)
                        action = pred_action[:, i]

                        total_rewards.append(rewards[0])
                        total_height_rewards.append(rewards[1])
                        total_dones.append(done)

                        if done.any():
                            ws.reset(done)

                        if real_time:
                            delta = time.time() - start
                            if delta < 0.04 and real_time:
                                time.sleep(0.04 - delta)
                            start = time.time()

                        t += 1
                        pbar.update(1)

        total_rewards = np.array(total_rewards).T
        total_height_rewards = np.array(total_height_rewards).T
        total_dones = np.array(total_dones).T

        # split rewards by lambda
        if self.cond_mask_prob > 0:
            for i, lam in enumerate(lambda_values):
                return_dict[f"lamda_{lam}/reward_mean"] = total_rewards[
                    i * self.envs_per_lambda : (i + 1) * self.envs_per_lambda
                ].mean()
                return_dict[f"lamda_{lam}/reward_std"] = total_rewards[
                    i * self.envs_per_lambda : (i + 1) * self.envs_per_lambda
                ].std()
                return_dict[f"lamda_{lam}/height_reward_mean"] = total_height_rewards[
                    i * self.envs_per_lambda : (i + 1) * self.envs_per_lambda
                ].mean()
                return_dict[f"lamda_{lam}/terminals_mean"] = total_dones[
                    i * self.envs_per_lambda : (i + 1) * self.envs_per_lambda
                ].mean()

            # compute the max reward mean
            reward_means = {
                k: v for k, v in return_dict.items() if k.endswith("/reward_mean")
            }
            max_reward_mean = max(reward_means.values())
            return_dict["max_reward_mean"] = max_reward_mean

            # reset cond_lambda
            ws.agent.model.cond_lambda = prev_lambda
        else:
            return_dict["reward_mean"] = total_rewards.mean()
            return_dict["terminals_mean"] = total_dones.mean()

        return return_dict

    #########################
    # Environment interface #
    #########################

    def step(self, action):
        self.env.step(action, self._done)
        obs, vel_cmd = self.observe()
        rewards = self.compute_reward(obs, vel_cmd)
        return obs, vel_cmd, rewards, self._done.copy()

    def observe(self, update_statistics=False):
        self.env.observe(self._observation, update_statistics)
        obs_and_cmd = self._observation[:, :36]
        obs_and_cmd = torch.from_numpy(obs_and_cmd).to(self.device)
        obs = obs_and_cmd[:, :33]
        # vel_cmd = obs_and_cmd[:, 33:36]
        vel_cmd = self.vel_cmd
        return obs, vel_cmd

    def reset(self, conditional_reset=False):
        if not conditional_reset:
            self.env.reset()
        else:
            self.env.conditionalReset()
            return self.env.conditionalResetFlags()

        return self.observe()

    def get_base_position(self):
        self.env.getBasePosition(self._base_position)
        return self._base_position

    def get_base_orientation(self):
        self.env.getBaseOrientation(self._base_orientation)
        return self._base_orientation

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def turn_on_visualization(self):
        self.env.turnOnVisualization()

    def turn_off_visualization(self):
        self.env.turnOffVisualization()

    ####################
    # Helper functions #
    ####################

    def set_lambdas(self, lambda_values: list):
        """
        Set lambda to a tensor of values for parallel evaluation
        """
        if len(lambda_values) == 0:
            lambda_values = self.lambda_values

        assert self.num_envs % len(lambda_values) == 0

        lambda_tensor = torch.zeros(self.num_envs, 1, 1).to(self.device)
        self.envs_per_lambda = self.num_envs // len(lambda_values)

        for i, lam in enumerate(lambda_values):
            lambda_tensor[i * self.envs_per_lambda : (i + 1) * self.envs_per_lambda] = (
                lam
            )

        return lambda_tensor, lambda_values

    def set_skill(self, idx: int):
        self.skill.fill_(0)
        self.skill[:, idx] = 1

    def compute_reward(self, obs: torch.Tensor, vel_cmds: torch.Tensor):
        rewards = reward_function(obs, vel_cmds, self.reward_fn)
        rewards = rewards.cpu().numpy()

        # height reward
        height = torch.from_numpy(self.get_base_position()[:, -1])
        if self.skill[0, 0] == 1:
            height_reward = torch.exp(-100 * (height - 0.6).pow(2))
        elif self.skill[0, 1] == 1:
            height_reward = torch.exp(-100 * (height - 0.5).pow(2))
        else:
            height_reward = torch.zeros_like(height)

        return rewards, height_reward.cpu().numpy()
