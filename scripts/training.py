import random
import sys

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # debug mode
    if sys.gettrace() is not None:
        cfg.wandb_mode = "disabled"
    else:
        cfg.wandb_mode = "online"

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    workspace = hydra.utils.instantiate(cfg)
    workspace.train()

    wandb.finish()
    print("Training done!")


if __name__ == "__main__":
    # Disable Hydra directory creation in debug mode
    if sys.gettrace() is not None:
        sys.argv.append("hydra.output_subdir=null")
        sys.argv.append("hydra.run.dir=.")
    main()
