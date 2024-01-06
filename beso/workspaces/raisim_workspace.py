import logging
import os 
import time

from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import hydra
from tqdm import tqdm
import wandb

from beso.workspaces.base_workspace_manager import BaseWorkspaceManger
from beso.networks.scaler.scaler_class import MinMaxScaler
from beso.envs.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from beso.envs.utils import get_split_idx
from beso.envs.raisim.data.dataloader import RaisimTrajectoryDataset
from beso.agents.diffusion_agents.beso_agent import BesoAgent
from beso.envs.raisim.raisim_env import RaisimEnv

log = logging.getLogger(__name__)


class RaisimManager(BaseWorkspaceManger):
    def __init__(
            self,
            seed: int,
            device: str,
            dataset_fn: DictConfig,
            goal_fn: DictConfig,
            eval_n_times: int,
            eval_n_steps,
            scale_data: bool,
            render: bool,
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
        self.render = render
        # get goal function for evaluation
        # self.goals_fn = hydra.utils.call(goal_fn)
        self.mask_obs = dataset_fn.transform.mask_targets if hasattr(dataset_fn.transform, 'mask_targets') else False
        self.reduce_obs_dim = dataset_fn.transform.reduce_obs_dim if hasattr(dataset_fn.transform, 'reduce_obs_dim') else False
        self.raisim_traj = RaisimTrajectoryDataset(
            goal_fn.data_path, onehot_goals=True
        )
        self.goal_conditional = dataset_fn.goal_conditional

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
        cfg,
        env_cfg,
        evaluate_multigoal: bool = True, # just for for same input as kitchen environment
        evaluate_sequential: bool = True, # just for for same input as kitchen environment
        log_wandb: bool = True, 
        new_sampler_type=None,
        n_inference_steps=None,
        get_mean=None,
        extra_args=None, 
        noise_scheduler=None,
        store_video=False,
        video_path=None  # Path to store the generated video
        ):
        """
        Test the agent on the environment with the given goal function
        """
        
        resource_dir = os.path.dirname(os.path.realpath(__file__)) + "/../envs/raisim/resources"
        env_cfg = OmegaConf.to_yaml(env_cfg)
        self.env = RaisimEnv(resource_dir, env_cfg)
        log.info('Starting trained model evaluation on the multimodal blockpush environment')
        rewards = []
        for goal_idx in range(self.eval_n_times):
            total_reward = 0
            done = False
            obs = self.env.reset()
            obs = torch.from_numpy(obs).to(cfg.device)
            goal = torch.zeros_like(obs)

            # now run the agent for n steps 
            for n in tqdm(range(self.eval_n_steps)):
                start = time.time()
                if done or n == self.eval_n_steps-1:
                    rewards.append(total_reward)
                    print('Total reward: {}'.format(total_reward))
                    if log_wandb:
                        wandb.log({ 'Reward': total_reward })
                    break

                if isinstance(agent, BesoAgent):
                    infer_start = time.time()
                    pred_action = agent.predict(
                        {'observation': obs,
                         'goal_observation': goal}, 
                        new_sampler_type=new_sampler_type,
                        new_sampling_steps=n_inference_steps,
                        get_mean=get_mean,
                        extra_args={}, 
                        noise_scheduler=noise_scheduler,
                    )
                    infer_end = time.time()
                    print(f"Inference time: {infer_end - infer_start}")
                else:
                    sampler_dict = {}
                    if n_inference_steps is not None:
                        sampler_dict['num_sampling_steps'] = n_inference_steps 
                    pred_action = agent.predict(
                        {'observation': obs,
                         'goal_observation': goal}, 
                    )
                if len(pred_action.shape) == 3:
                    pred_action = pred_action.squeeze(0)
                obs, reward, done = self.env.step(pred_action.detach().cpu().numpy())
                obs = torch.from_numpy(obs).to(cfg.device)
                total_reward += reward
                # get current goal
                # if self.goal_conditional == "onehot":
                #     goal = self.goals_fn(obs, goal_idx, n)
                
                delta = time.time() - start
                if delta < 0.02:
                    time.sleep(0.02 - delta)
                
        log.info(f"Total reward: {total_reward}")
        
        self.env.close()

        avrg_reward = sum(rewards)/len(rewards)
        std_reward = np.array(rewards).std()
        
        print('average reward {}'.format(avrg_reward))
        log.info('... finished trained model evaluation of the blockpush environment environment.')
        if log_wandb:
            wandb.log({"Average_reward": avrg_reward})
        if log_wandb:
            log.info(f"---------------------------------------")
        else:
            print("---------------------------------------")
        return_dict = {
            'avrg_reward': avrg_reward,
            'std_reward': std_reward,
        }
        # return the average reward 
        return return_dict
    
    def _report_result_upon_completion(self, goal_idx=None):
        """
        Report the result upon completion of the episode
        """
        if goal_idx is not None:
            train_idx, val_idx = get_split_idx(
                len(self.push_traj),
                seed=self.seed,
                train_fraction=self.train_fraction,
            )
            _, _, _, onehot_goals = self.push_traj[train_idx[goal_idx]]
            onehot_mask, first_frame = onehot_goals.max(0)
            goals = [(first_frame[i], i) for i in range(4) if onehot_mask[i]]
            goals = sorted(goals, key=lambda x: x[0])
            goals = [g[1] for g in goals]
            logging.info(f"Expected tasks {goals}")
            expected_tasks = set(goals)
            conditional_done = set(self.env.all_completions).intersection(
                expected_tasks
            )
            return len(conditional_done) / 2
        else:
            return len(self.env.all_completions) / 2
