import logging
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from env.raisim_env import RaisimEnv

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
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
    cfg["obs_dim"] = 38
    cfg["pred_obs_dim"] = 35

    # init wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    wandb.init(
        project=cfg.wandb.project, mode=mode, config=wandb.config, dir=output_dir
    )

    agent = hydra.utils.instantiate(cfg.agents)
    agent.env = RaisimEnv(cfg)
    agent.working_dir = output_dir

    agent.train_agent()

    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    main()
