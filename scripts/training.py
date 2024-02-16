import os
import logging
import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)


@hydra.main(config_path="../configs", config_name="raisim_main_config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    
    # set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # debug mode
    if sys.gettrace() is not None:
        mode = "disabled"
        cfg["sim_every_n_steps"] = 10
    else:
        mode = "online"

    # set the observation dimension
    if cfg["data_path"] == "rand_feet":
        cfg["obs_dim"] = 48
    elif cfg["data_path"] == "rand_feet_com":
        cfg["obs_dim"] = 59
    elif cfg["data_path"].startswith("fwd"):
        cfg["obs_dim"] = 33

    # init wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    wandb.init(
        project=cfg.wandb.project, 
        mode=mode,
        config=wandb.config,
        dir=output_dir
    )

    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)

    # get the scaler instance and set the bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.working_dir = output_dir

    # initialize the environment
    workspace_manager.init_env(cfg, cfg.data_path)

    # train
    agent.train_agent(
        workspace_manager.data_loader['train'],
        workspace_manager.data_loader['test'],
        workspace_manager.test_agent,
    )

    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    main()
