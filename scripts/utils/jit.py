import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_path = "logs/2024-09-12/21-26-05/"

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(model_path)
    agent.model.eval()
    agent.model.inner_model.detach_all()

    obs = torch.zeros(1, 8, 33)
    vel_cmd = torch.zeros(1, 3)
    skill = torch.zeros(1, 2)
    returns = torch.zeros(1, 1)
    scripted_agent = torch.jit.trace(
        agent.forward, (obs, vel_cmd, skill, returns), check_trace=False
    )

    scripted_agent.save(f"data/models/policy_{agent.device}.pt")
    print(f"Scripted agent saved to policy_{agent.device}.pt")


if __name__ == "__main__":
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    main()
