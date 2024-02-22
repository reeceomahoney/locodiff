import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from beso.agent.diffusion_agents.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()

@hydra.main(config_path="../configs", config_name="evaluate_raisim.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    
    if cfg.log_wandb:
        wandb.init(
            project='beso_eval', 
            mode="disabled",  
            config=wandb.config
        )

    # config
    cfg_store_path = os.path.join(os.getcwd(), cfg.model_store_path,'.hydra/config.yaml')
    model_cfg = OmegaConf.load(cfg_store_path) 
    model_cfg.device = cfg.device
    model_cfg.workspaces['device'] = cfg['device']
    model_cfg.agents['device'] = cfg['device']
    model_cfg.env['num_envs'] = 1
    model_cfg.env['server_port'] = 8081

    # set the observation dimension
    if model_cfg["data_path"] == "rand_feet":
        model_cfg["obs_dim"] = 48
    elif model_cfg["data_path"] == "rand_feet_com":
        model_cfg["obs_dim"] = 59
    elif model_cfg["data_path"].startswith("fwd"):
        model_cfg["obs_dim"] = 33

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    workspace_manager = hydra.utils.instantiate(model_cfg.workspaces)
    agent = hydra.utils.instantiate(model_cfg.agents)

    # get the scaler instance and set the bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.load_pretrained_model(cfg.model_store_path)

    # set new noise limits
    agent.sigma_max = cfg.sigma_max
    agent.sigma_min = cfg.sigma_min

    # initialize the environment
    workspace_manager.init_env(model_cfg, model_cfg.data_path)

    # set seeds
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if cfg['test_classifier_free_guidance']:
        # load the classifier free wrapper
        agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=cfg['cond_lambda'])
    elif cfg['cond_lambda'] > 1:
        agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=cfg['cond_lambda'])

    # test prediction accuracy of model on ground truth data
    test_timestep_mse = False
    if test_timestep_mse:
        dataloader = workspace_manager.make_dataloaders()["test"]
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        inference_steps = [1, 2, 3, 4, 5, 10, 20, 40, 50]
        results = []

        for step in inference_steps:
            agent.num_sampling_steps = step
            info = agent.evaluate(batch)
            results.append(info['timestep_mse'].cpu().numpy())
        
        for i, result in enumerate(results):
            plt.plot(np.arange(1,21), result, label=f'{inference_steps[i]} inference steps')

        plt.yscale('log')
        plt.legend()
        plt.savefig('timestep_mse.png')
    
    test_rollout = True
    if test_rollout:
        workspace_manager.eval_n_times = cfg['num_runs']
        results_dict = workspace_manager.test_agent(
            agent, n_inference_steps=cfg['n_inference_steps'], real_time=True
        )
        print(results_dict)


if __name__ == "__main__":
    main()