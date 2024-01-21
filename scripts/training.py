import os
import logging
import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np


from beso.agents.diffusion_agents.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel
from beso.agents.diffusion_agents.beso_agent import BesoAgent

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)


@hydra.main(config_path="../configs", config_name="raisim_main_config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # init wandb logger and config from hydra path 
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    debugging = sys.gettrace() is not None
    mode = "disabled" if debugging else "online"

    run = wandb.init(
        project=cfg.wandb.project, 
        # entity=cfg.wandb.entity,
        # group=cfg.group,
        mode=mode,
        config=wandb.config,
        dir=output_dir
    )

    # load the required classes to train and test the agent
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)
    agent.working_dir = output_dir

    agent.train_agent(
        workspace_manager.data_loader['train'],
        workspace_manager.data_loader['test']
    )

    # after training, test the agent
    use_feet_pos = True if cfg.data_path == 'rand_feet' else False
    if isinstance(agent, BesoAgent):
        if cfg.cond_mask_prob > 0:
            agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=cfg.cond_lambda, obs_dim=cfg.obs_dim)
            log.info(f'using cond lambda_value of {cfg.cond_lambda}')
            result_dict = workspace_manager.test_agent(
                agent,
                cfg,
                cfg.env,
                log_wandb=True,
                use_feet_pos=use_feet_pos,
                )
        else:
            result_dict = workspace_manager.test_agent(
                agent,
                cfg.evaluate_multigoal,
                cfg.evaluate_sequential,
                log_wandb=True,
                )
    else:
        result_dict = workspace_manager.test_agent(
            agent,
            cfg.evaluate_multigoal,
            cfg.evaluate_sequential,
            log_wandb=True
            )
    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    main()
