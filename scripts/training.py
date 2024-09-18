import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

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
