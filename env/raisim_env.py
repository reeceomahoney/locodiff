import logging
import os
import platform
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

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

        # get environment information
        self.num_obs = self.env.getObDim()
        self.num_acts = self.env.getActionDim()
        self.T_action = cfg.T_action
        self.eval_n_times = cfg.env.eval_n_times
        self.eval_n_steps = cfg.env.eval_n_steps
        self.device = cfg.device
        self.dataset = cfg.data_path

        # initialize variables
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._frame_cartesian_pos = np.zeros([self.num_envs, 3 * 5], dtype=np.float32)
        self._base_orientation = np.zeros([self.num_envs, 9], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        
        self.goal = None

    def step(self, action):
        self.env.step(action, self._reward, self._done)
        return *self.observe(), self._reward.copy(), self._done.copy()

    def observe(self, update_statistics=False):
        self.env.observe(self._observation, update_statistics)

        base_pos = self.get_frame_cartesian_pos()[:, :2]
        obs = np.concatenate([base_pos, self._observation[:, :33]], axis=-1)
        cmd = self.goal - base_pos

        obs = torch.from_numpy(obs).to(self.device)
        cmd = torch.from_numpy(cmd).to(self.device)

        return obs, cmd

    def reset(self, conditional_reset=False):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)

        if not conditional_reset:
            self.env.reset()
        else:
            self.env.conditionalReset()
            return self.env.conditionalResetFlags()

        return self.observe()

    def close(self):
        self.env.close()

    def simulate(
        self,
        agent,
        n_inference_steps=None,
        real_time=False,
    ):
        """
        Test the agent on the environment with the given goal function
        """
        log.info("Starting trained model evaluation")

        total_rewards = np.zeros(self.num_envs, dtype=np.float32)
        total_dones = np.zeros(self.num_envs, dtype=np.int64)
        agent.reset()  # TODO: this is incorrect, needs to reset episodes separately
        self.env.reset()
        self.generate_goal()

        for _ in range(self.eval_n_times):
            done = np.array([False])
            obs, cmd = self.observe()

            # now run the agent for n steps
            for n in tqdm(range(self.eval_n_steps)):
                start = time.time()

                if done.any():
                    total_dones += done
                if n == self.eval_n_steps - 1:
                    total_dones += np.ones(done.shape, dtype="int64")

                pred_action = agent.predict(
                    {"observation": obs, "cmd": cmd},
                    new_sampling_steps=n_inference_steps,
                )
                pred_action = pred_action.detach().cpu().numpy()

                for i in range(self.T_action):
                    obs, cmd, reward, done = self.step(pred_action[:, i])
                    reward = cmd.norm(dim=-1).cpu().numpy()
                    total_rewards += reward

                    if not n % 250:
                        self.generate_goal()

                    delta = time.time() - start
                    if delta < 0.04 and real_time:
                        time.sleep(0.04 - delta)
                    start = time.time()

        self.close()
        total_rewards /= total_dones
        avrg_reward = total_rewards.mean()
        std_reward = total_rewards.std()

        log.info("... finished trained model evaluation")
        return_dict = {
            "avrg_reward": avrg_reward,
            "std_reward": std_reward,
            "total_done": total_dones.mean(),
        }
        return return_dict

    def generate_goal(self):
        self.goal = np.random.uniform(-4, 4, (self.num_envs, 2)).astype(np.float32)
        self.set_goal(self.goal)

    def get_frame_cartesian_pos(self):
        self.env.getFrameCartesianPositions(self._frame_cartesian_pos)
        return self._frame_cartesian_pos

    def get_base_orientation(self):
        self.env.getBaseOrientation(self._base_orientation)
        return self._base_orientation

    def kill_server(self):
        self.env.killServer()

    def set_goal(self, goal):
        self.env.setGoal(goal)

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def turn_on_visualization(self):
        self.env.turnOnVisualization()

    def turn_off_visualization(self):
        self.env.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.env.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.env.stopRecordingVideo()

    @property
    def num_envs(self):
        return self.env.getNumOfEnvs()
