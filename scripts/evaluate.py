import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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
    model_cfg.env["server_port"] = cfg.server_port
    model_cfg.env["max_time"] = 100
    model_cfg.T_action = 1
    model_cfg.use_ema = False
    model_cfg.evaluating = True
    model_cfg.env["num_envs"] = 25 * len(cfg.lambda_values)
    model_cfg.n_timesteps = cfg.n_inference_steps
    model_cfg.agents.output_dir = cfg.model_store_path

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

    # Evaluate
    if cfg["test_rollout"]:
        env.eval_n_times = cfg["num_runs"]
        results_dict = env.simulate(agent, real_time=True)
        print(results_dict)
    if cfg["test_reward_lambda"]:
        results_dict = env.simulate(
            agent, real_time=False, lambda_values=cfg.lambda_values
        )
        returns = [v for k, v in results_dict.items() if k.endswith("/return_mean")]
        terminals = [
            v for k, v in results_dict.items() if k.endswith("/terminals_mean")
        ]

        print(returns)
        print(terminals)
        plt.bar(range(len(returns)), returns)
        plt.xticks(range(len(cfg.lambda_values)), cfg.lambda_values)
        plt.xlabel("Lambda")
        plt.ylabel("Velocity tracking return")
        plt.show()
    if cfg["test_mse"]:
        dataloader = agent.test_loader
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        info = agent.evaluate(batch)
        print(info["total_mse"])


if __name__ == "__main__":
    main()
