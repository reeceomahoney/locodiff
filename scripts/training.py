import logging
import sys

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

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
        cfg["num_hidden_layers"] = 1
        output_dir = "/tmp"
    else:
        mode = "online"
        output_dir = HydraConfig.get().runtime.output_dir

    # init wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project, mode=mode, config=wandb.config, dir=output_dir
    )

    cfg.agents.output_dir = output_dir
    agent = hydra.utils.instantiate(cfg.agents)
    agent.env = RaisimEnv(cfg)
    agent.working_dir = output_dir

    agent.train_agent()

    log.info("done")
    wandb.finish()


if __name__ == "__main__":
    # Disable Hydra directory creation in debug mode
    if sys.gettrace() is not None:
        sys.argv.append("hydra.output_subdir=null")
        sys.argv.append("hydra.run.dir=.")
    main()
