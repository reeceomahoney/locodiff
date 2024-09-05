import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.manifold import TSNE

from env.raisim_env import RaisimEnv


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


@hydra.main(config_path="../configs", config_name="evaluate.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    # config
    cfg_store_path = os.path.join(
        os.getcwd(), cfg.model_store_path, ".hydra/config.yaml"
    )
    model_cfg = OmegaConf.load(cfg_store_path)
    model_cfg.device = cfg.device
    model_cfg.agents["device"] = cfg.device
    model_cfg.env["num_envs"] = cfg.num_envs
    model_cfg.env["server_port"] = cfg.server_port
    model_cfg.env["max_time"] = 100
    model_cfg["T_action"] = 1
    model_cfg["use_ema"] = False
    model_cfg["evaluating"] = True

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    agent = hydra.utils.instantiate(model_cfg.agents)
    agent.load_pretrained_model(cfg.model_store_path)
    # agent = torch.jit.load(f"data/models/policy_{cfg.device}.pt")
    env = RaisimEnv(model_cfg)

    # set new noise limits
    agent.sigma_max = cfg.sigma_max
    agent.sigma_min = cfg.sigma_min
    agent.cond_lambda = cfg.cond_lambda

    # Evaluate
    if cfg["test_rollout"]:
        env.eval_n_times = cfg["num_runs"]
        results_dict = env.simulate(agent, real_time=True)
        print(results_dict)
    if cfg["test_reward_lambda"]:
        # lambda_values = [0, 1, 2, 10, 20, 50, 100, 200, 500]
        lambda_values = [0, 1, 1.2, 1.5, 2]

        results_dict = env.simulate(agent, real_time=False, lambda_values=lambda_values)
        rewards = [v for k, v in results_dict.items() if k.endswith("/reward_mean")]
        terminals = [
            v for k, v in results_dict.items() if k.endswith("/terminals_mean")
        ]

        print(rewards)
        print(terminals)
        plt.bar(range(len(rewards)), rewards)
        plt.xticks(range(len(lambda_values)), lambda_values)
        plt.xlabel("Lambda")
        plt.ylabel("Velocity tracking reward")
        plt.savefig("results.png")
    if cfg["test_mse"]:
        dataloader = agent.test_loader
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        info = agent.evaluate(batch)
        print(info["total_mse"])


if __name__ == "__main__":
    main()
