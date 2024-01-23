import logging
import os 
import time

from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import hydra
from tqdm import tqdm

from beso.workspaces.base_workspace_manager import BaseWorkspaceManger
from beso.networks.scaler.scaler_class import MinMaxScaler, Scaler
from beso.envs.raisim.raisim_env import RaisimEnv

log = logging.getLogger(__name__)


class RaisimManager(BaseWorkspaceManger):
    def __init__(
            self,
            seed: int,
            device: str,
            dataset_fn: DictConfig,
            eval_n_times: int,
            eval_n_steps,
            goal_dim: int,
            scale_data: bool,
            train_batch_size: int = 256,
            test_batch_size: int = 256,
            num_workers: int = 4,
            train_fraction: float = 0.95,
            use_minmax_scaler: bool = False,
    ):
        super().__int__(seed, device)
        print("Using Raisim environment")
        self.eval_n_times = eval_n_times
        self.eval_n_steps = eval_n_steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.datasets = hydra.utils.call(dataset_fn)
        self.train_set, self.test_set = self.datasets
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.scale_data = scale_data
        self.use_minmax_scaler = use_minmax_scaler
        self.scaler = None
        self.data_loader = self.make_dataloaders()
        self.goal_dim = goal_dim
    
    def init_env(self, cfg, use_feet_pos=False):
        resource_dir = os.path.dirname(os.path.realpath(__file__)) + "/../envs/raisim/resources"
        env_cfg = OmegaConf.to_yaml(cfg.env)
        self.env = RaisimEnv(resource_dir, env_cfg, use_feet_pos)

    def make_dataloaders(self):
        """
        Create a training and test dataloader using the dataset instances of the task
        """
        if self.use_minmax_scaler:
            self.scaler = MinMaxScaler(self.train_set.dataset.dataset.get_all_observations(), self.train_set.dataset.dataset.get_all_actions(), self.scale_data, self.device)
        else:
            self.scaler = Scaler(self.train_set.dataset.dataset.get_all_observations(), self.train_set.dataset.dataset.get_all_actions(), self.scale_data, self.device)
        
        train_dataloader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return {
            "train": train_dataloader,
            "test": test_dataloader
        }

    def test_agent(
        self, 
        agent, 
        n_inference_steps=None,
        ):
        """
        Test the agent on the environment with the given goal function
        """
        log.info('Starting trained model evaluation')
        total_rewards = 0
        total_dones = 0
        obs = self.env.reset()
        goal = torch.zeros(obs.shape[0], 1, self.goal_dim).to(self.device)
        for _ in range(self.eval_n_times):
            done = np.array([False])
            obs = self.env.observe()
            obs = torch.from_numpy(obs).to(self.device)

            # now run the agent for n steps 
            for n in tqdm(range(self.eval_n_steps)):
                start = time.time()

                if done.any():
                    total_dones += done
                if n == self.eval_n_steps-1:
                    total_dones += np.ones(done.shape, dtype='int64')

                pred_action = agent.predict(
                    {'observation': obs, 'goal': goal}, 
                    new_sampling_steps=n_inference_steps,
                )
                obs, reward, done = self.env.step(pred_action.detach().cpu().numpy())
                obs = torch.from_numpy(obs).to(self.device)
                total_rewards += reward
            
                delta = time.time() - start
                if delta < 0.02:
                    time.sleep(0.02 - delta)
                
        self.env.close()
        total_rewards /= total_dones
        avrg_reward = total_rewards.mean()
        std_reward = total_rewards.std()
        
        log.info('... finished trained model evaluation')
        return_dict = {
            'avrg_reward': avrg_reward,
            'std_reward': std_reward,
        }
        return return_dict
    