import logging
import os
import platform
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from env.lib.raisim_env import RaisimWrapper
from locodiff.utils import reward_function

log = logging.getLogger(__name__)


class RaisimEnv:

    def __init__(self, cfg, seed=0):
        if platform.system() == "Darwin":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        resource_dir = os.path.dirname(os.path.realpath(__file__)) + "/resources"
        env_cfg = OmegaConf.to_yaml(cfg.env)

        # initialize environment
        self.env = RaisimWrapper(resource_dir, env_cfg)
        self.env.setSeed(seed)
        self.env.turnOnVisualization()

        # get environment information
        self.num_obs = self.env.getObDim()
        self.num_acts = self.env.getActionDim()
        self.T_action = cfg.T_action
        self.skill_dim = cfg.skill_dim
        self.window = cfg.T_cond + cfg.T - 1
        self.eval_n_times = cfg.env.eval_n_times
        self.eval_n_steps = cfg.env.eval_n_steps
        self.device = cfg.device
        self.dataset = cfg.data_path

        # initialize variables
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._base_position = np.zeros([self.num_envs, 3], dtype=np.float32)
        self._base_orientation = np.zeros([self.num_envs, 4], dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._torques = np.zeros([self.num_envs, 12], dtype=np.float32)

        self.nominal_joint_pos = np.zeros([self.num_envs, 12], dtype=np.float32)
        self.env.getNominalJointPositions(self.nominal_joint_pos)

        self.reward_fn = cfg.reward_fn

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

    def simulate(self, agent, real_time=False, lambda_values=None):
        log.info("Starting trained model evaluation")

        returns = torch.ones((self.num_envs, 1)).to(self.device)
        self.skill = torch.zeros(self.num_envs, self.skill_dim).to(self.device)
        self.skill[:, 0] = 1

        self.vel_cmd = torch.randint(
            0, 2, (self.num_envs, 1), device=self.device
        ).float()

        cond_lambdas = lambda_values if lambda_values is not None else [0, 1, 2, 5]
        assert self.num_envs % len(cond_lambdas) == 0

        # For parallel evaluation
        lambda_tensor = torch.zeros(self.num_envs, 1, 1).to(self.device)
        envs_per_lambda = self.num_envs // len(cond_lambdas)
        for i, lam in enumerate(cond_lambdas):
            lambda_tensor[i * envs_per_lambda : (i + 1) * envs_per_lambda] = lam
        agent.cond_lambda = lambda_tensor

        return_dict = {}

        for _ in range(self.eval_n_times):
            total_rewards = np.zeros(self.num_envs, dtype=np.float32)
            height_rewards = np.zeros(self.num_envs, dtype=np.float32)
            total_dones = np.zeros(self.num_envs, dtype=np.int64)

            action = self.nominal_joint_pos
            done = np.array([False])

            self.env.reset()
            agent.reset(np.ones(self.num_envs, dtype=bool))

            obs, vel_cmd = self.observe()

            for n in tqdm(range(self.eval_n_steps)):
                start = time.time()

                if done.any():
                    total_dones += done
                    agent.reset(done)

                if n == 125:
                    self.skill = torch.zeros(self.num_envs, self.skill_dim).to(
                        self.device
                    )
                    self.skill[:, 1] = 1

                pred_action = agent.predict(
                    {
                        "obs": obs,
                        "skill": self.skill,
                        "vel_cmd": vel_cmd,
                        "return": returns,
                    },
                )

                for i in range(self.T_action):
                    obs, vel_cmd, rewards, done = self.step(action)
                    total_rewards += rewards[0]
                    height_rewards += rewards[1]
                    action = pred_action[:, i]

                    if real_time:
                        delta = time.time() - start
                        if delta < 0.04 and real_time:
                            time.sleep(0.04 - delta)
                        start = time.time()

            # split rewards by lambda
            total_rewards /= self.eval_n_steps
            height_rewards /= self.eval_n_steps
            for i, lam in enumerate(cond_lambdas):
                return_dict[f"lamda_{lam}/reward_mean"] = total_rewards[
                    i * envs_per_lambda : (i + 1) * envs_per_lambda
                ].mean()
                return_dict[f"lamda_{lam}/reward_std"] = total_rewards[
                    i * envs_per_lambda : (i + 1) * envs_per_lambda
                ].std()
                return_dict[f"lamda_{lam}/terminals_mean"] = total_dones[
                    i * envs_per_lambda : (i + 1) * envs_per_lambda
                ].mean()
                return_dict[f"lamda_{lam}/height_reward_mean"] = height_rewards[
                    i * envs_per_lambda : (i + 1) * envs_per_lambda
                ].mean()

        return return_dict

    def compute_reward(self, obs, vel_cmds):
        rewards = reward_function(obs, vel_cmds, self.reward_fn)
        rewards = rewards.cpu().numpy()

        # height reward
        height = torch.from_numpy(self.get_base_position()[:, -1])
        if self.skill[0, 0] == 1:
            height_reward = torch.exp(-100 * (height - 0.6).pow(2))
        elif self.skill[0, 1] == 1:
            height_reward = torch.exp(-100 * (height - 0.5).pow(2))
        height_reward = height_reward.cpu().numpy()

        return rewards, height_reward

    def get_base_position(self):
        self.env.getBasePosition(self._base_position)
        return self._base_position

    def get_base_orientation(self):
        self.env.getBaseOrientation(self._base_orientation)
        return self._base_orientation

    def set_goal(self, goal):
        self.env.setGoal(goal)

    def get_torques(self):
        self.env.getTorques(self._torques)
        return torch.from_numpy(self._torques).to(self.device)

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def turn_on_visualization(self):
        self.env.turnOnVisualization()

    def turn_off_visualization(self):
        self.env.turnOffVisualization()

    @property
    def num_envs(self):
        return self.env.getNumOfEnvs()
