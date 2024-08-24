import logging
import os
import platform
import time

import imageio
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt

from env.lib.raisim_env import RaisimWrapper

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

        self.goal = None

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

        returns = torch.ones((self.num_envs, self.window, 1)).to(self.device)
        self.skill = torch.zeros(self.num_envs, self.skill_dim).to(self.device)
        self.skill[:, 0] = 1
        self.images = []

        self.vel_cmd = torch.randint(0, 2, (self.num_envs, 1)).to(self.device)
        self.vel_cmd = self.vel_cmd.float() * 2 - 1

        if lambda_values is None:
            cond_lambdas = [0, 1, 2, 5, 10]
        else:
            cond_lambdas = lambda_values
        return_dict = {}

        for lam in cond_lambdas:
            total_rewards = np.zeros(self.num_envs, dtype=np.float32)
            total_dones = np.zeros(self.num_envs, dtype=np.int64)
            agent.cond_lambda = lam

            self.env.reset()
            agent.reset()
            done = np.array([False])
            obs, vel_cmd = self.observe()

            # now run the agent for n steps
            action = self.nominal_joint_pos
            action = np.tile(action, (self.num_envs, 1))
            for n in tqdm(range(self.eval_n_steps)):
                start = time.time()

                if done.any():
                    total_dones += done
                    agent.reset()
                if n == self.eval_n_steps - 1:
                    total_dones += np.ones(done.shape, dtype="int64")

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
                    total_rewards += rewards
                    action = pred_action[:, i]

                    if real_time:
                        delta = time.time() - start
                        if delta < 0.04 and real_time:
                            time.sleep(0.04 - delta)
                        start = time.time()

            total_rewards /= self.eval_n_steps
            avrg_reward = total_rewards.mean()
            std_reward = total_rewards.std()

            return_dict[f"lamda_{lam}/reward_mean"] = avrg_reward
            return_dict[f"lamda_{lam}/reward_std"] = std_reward
            return_dict[f"lamda_{lam}/terminals_mean"] = total_dones.mean()

        return return_dict

    def compute_reward(self, obs, vel_cmds):
        # velocity reward
        lin_vel = obs[:, 30:32]
        ang_vel = obs[:, 17:18]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)

        vel = vel[:, 0]
        vel_cmds = vel_cmds.squeeze()

        # rewards = torch.zeros_like(vel)
        # rewards = torch.where(vel_cmds == 1, vel, rewards)
        # rewards = torch.where(vel_cmds == -1, -vel, rewards)
        rewards = vel
        rewards = rewards.cpu().numpy()

        return rewards

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
