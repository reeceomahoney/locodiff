import os
import logging
import numpy as np

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from beso.agents.diffusion_agents.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel


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

    # set the observation dimension
    if model_cfg["data_path"] == "rand_feet":
        model_cfg["obs_dim"] = 48
    elif model_cfg["data_path"] == "rand_feet_com":
        model_cfg["obs_dim"] = 59

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    workspace_manager = hydra.utils.instantiate(model_cfg.workspaces)
    agent = hydra.utils.instantiate(model_cfg.agents)

    # get the scaler instance and set the bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)
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

    if cfg.test_single_variant:
        workspace_manager.eval_n_times = cfg['num_runs']
        results_dict = workspace_manager.test_agent(agent, n_inference_steps=cfg['n_inference_steps'])
        print(results_dict)
    if cfg.test_all_samplers:
        workspace_manager.compare_sampler_types(
            agent, 
            num_runs=cfg['num_runs'], 
            num_steps_per_run=280, 
            log_wandb=False, 
            n_inference_steps=cfg['n_inference_steps'],
            get_mean=None,
            store_path=cfg.model_store_path
        )
    if cfg.compare_samplers_over_diffent_steps:
        workspace_manager.compare_sampler_types_over_n_steps(
            agent, 
            num_runs=cfg['num_runs'],  
            steps_list=  [3, 4, 5, 10, 20, 40, 50], #
            log_wandb=False, 
            get_mean=None,
            store_path=cfg.model_store_path,
            extra_args={
                's_churn': cfg['s_churn'],
                's_min': cfg['s_min']
                }
        )

    if cfg.compare_classifier_free_guidance:
        workspace_manager.compare_classifier_free_guidance(
            agent, 
            num_runs=cfg['num_runs'], 
            sampler_type=cfg['sampler_type'],
            num_steps_per_run=280,
            cond_lambda_list=[0, 1, 1.5, 2, 2.5], 
            n_inference_steps=cfg['n_inference_steps'],
            log_wandb=False, 
            get_mean=None,
            store_path=cfg.model_store_path,
            extra_args={
                's_churn': cfg['s_churn'],
                's_min': cfg['s_min']
                }
        )
    if cfg.compare_noisy_sampler:
        workspace_manager.compare_noisy_sampler(
            agent,  
            num_runs=cfg['num_runs'],
            num_steps_per_run=280,
            n_inference_steps=cfg['n_inference_steps'],
            log_wandb=False,
            get_mean=None,
            store_path=cfg.model_store_path
            )


if __name__ == "__main__":
    main()