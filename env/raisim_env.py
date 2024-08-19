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
        vel_cmd = obs_and_cmd[:, 33:36]
        return obs, vel_cmd

    def reset(self, conditional_reset=False):
        if not conditional_reset:
            self.env.reset()
        else:
            self.env.conditionalReset()
            return self.env.conditionalResetFlags()

        return self.observe()

    def simulate(self, agent, n_inference_steps=None, real_time=False):
        log.info("Starting trained model evaluation")

        total_rewards = np.zeros(self.num_envs, dtype=np.float32)
        height_rewards = np.zeros(self.num_envs, dtype=np.float32)
        total_dones = np.zeros(self.num_envs, dtype=np.int64)
        returns = torch.ones((self.num_envs, self.window, 1)).to(self.device)
        self.images = []

        for _ in range(self.eval_n_times):
            self.env.reset()
            agent.reset()
            done = np.array([False])
            obs, vel_cmd = self.observe()

            self.skill = torch.zeros(self.num_envs, self.skill_dim).to(self.device)
            self.skill[:, 0] = 1

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
                    new_sampling_steps=n_inference_steps,
                )

                for i in range(self.T_action):
                    obs, vel_cmd, rewards, done = self.step(action)
                    total_rewards += rewards[0]
                    height_rewards += rewards[1]
                    action = pred_action[:, i]

                    delta = time.time() - start
                    if delta < 0.04 and real_time:
                        time.sleep(0.04 - delta)
                    start = time.time()

        total_rewards /= self.eval_n_times * self.eval_n_steps
        avrg_reward = total_rewards.mean()
        std_reward = total_rewards.std()

        height_rewards /= self.eval_n_times * self.eval_n_steps
        avrg_height_reward = height_rewards.mean()
        std_height_reward = height_rewards

        log.info("... finished trained model evaluation")
        return_dict = {
            "avrg_reward": avrg_reward,
            "std_reward": std_reward,
            "avrg_height_reward": avrg_height_reward,
            "total_done": total_dones.mean(),
        }
        return return_dict

    def plot_trajectory(self, pred_traj, goal):
        # Calculate yaw angles from quaternions
        quat = pred_traj[0, :, 2:6]
        quat = np.roll(quat, shift=-1, axis=1)  # w, x, y, z -> x, y, z, w
        yaw = R.from_quat(quat).as_euler("xyz")[:, 2]

        goal = goal.cpu().numpy()

        # plot the trajectory
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.plot(pred_traj[0, :, 0], pred_traj[0, :, 1], "r")
        ax.plot(goal[0, 0], goal[0, 1], "go")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        # plot orientation using yaw angles
        for i in range(0, pred_traj.shape[1], 10):
            ax.quiver(
                pred_traj[0, i, 0],
                pred_traj[0, i, 1],
                pred_traj[0, i, 33],
                pred_traj[0, i, 34],
                color="b",
                scale=10,
                width=0.005,
            )

        # Convert plot to image array
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        self.images.append(image)

    def generate_goal(self):
        self.goal = np.random.uniform(-5, 5, (self.num_envs, 2)).astype(np.float32)
        self.set_goal(self.goal)
        self.goal = torch.from_numpy(self.goal).to(self.device)

    def compute_reward(self, obs, vel_cmd):
        # velocity reward
        lin_vel = obs[:, 30:32]
        ang_vel = obs[:, 17:18]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)
        reward = torch.exp(-(vel - vel_cmd).pow(2))
        reward = reward.mean(dim=-1).cpu().numpy()

        # height reward
        height = torch.from_numpy(self.get_base_position()[:, -1])
        if self.skill[0, 0] == 1:
            height_reward = torch.exp(-100 * (height - 0.6).pow(2))
        elif self.skill[0, 1] == 1:
            height_reward = torch.exp(-100 * (height - 0.5).pow(2))
        height_reward = height_reward.cpu().numpy()

        return reward, height_reward

    def get_base_position(self):
        self.env.getBasePosition(self._base_position)
        return self._base_position

    def get_base_orientation(self):
        self.env.getBaseOrientation(self._base_orientation)
        return self._base_orientation

    def set_goal(self, goal):
        self.env.setGoal(goal)

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def turn_on_visualization(self):
        self.env.turnOnVisualization()

    def turn_off_visualization(self):
        self.env.turnOffVisualization()

    @property
    def num_envs(self):
        return self.env.getNumOfEnvs()
