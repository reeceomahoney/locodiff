import numpy as np
import torch
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_path = "logs/2024-06-02/14-51-52"

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(model_path)

    observation = torch.zeros(1, 8, 36)
    cmd = torch.zeros(1, 3)
    indicator = torch.zeros(1, 2)
    scripted_agent = torch.jit.trace(agent.forward, (observation, cmd, indicator))

    scripted_agent.save(f"data/models/policy_{agent.device}.pt")
    print(f"Scripted agent saved to policy_{agent.device}.pt")


if __name__ == "__main__":
    main()
