import logging
import os
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from locodiff.classifier import ClassifierGuidedSampleModel

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
    model_cfg.env.impl.server_port = cfg.server_port
    model_cfg.env.impl.max_time = 100
    model_cfg.T_action = 1
    model_cfg.use_ema = False
    model_cfg.num_envs = 25 * len(cfg.lambda_values)
    model_cfg.env.impl.num_envs = 25 * len(cfg.lambda_values)
    model_cfg.sampling_steps = cfg.sampling_steps

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    workspace = hydra.utils.instantiate(model_cfg)
    workspace.load(cfg.model_store_path)
    # agent = torch.jit.load(f"data/models/policy_{cfg.device}.pt")

    # Evaluate
    if cfg["test_rollout"]:
        workspace.env.eval_n_times = cfg["num_runs"]
        results_dict = workspace.env.simulate(
            workspace, real_time=True, lambda_values=[0]
        )
        print(results_dict)
    if cfg["test_reward_lambda"]:
        results_dict = workspace.env.simulate(
            workspace, real_time=False, lambda_values=cfg.lambda_values
        )
        # returns = [v for k, v in results_dict.items() if k.endswith("/return_mean")]
        rewards = [v for k, v in results_dict.items() if k.endswith("/reward_mean")]
        rewards_std = [v for k, v in results_dict.items() if k.endswith("/reward_std")]
        terminals = [
            v for k, v in results_dict.items() if k.endswith("/terminals_mean")
        ]

        print(rewards)
        print(rewards_std)
        print(terminals)
        plt.bar(range(len(rewards)), rewards)
        plt.xticks(range(len(cfg.lambda_values)), cfg.lambda_values)
        plt.xlabel("Lambda")
        plt.ylabel("Velocity tracking return")
        plt.show()
    if cfg["test_mse"]:
        dataloader = workspace.test_loader
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        info = workspace.evaluate(batch)
        print(info["total_mse"])
    if cfg["test_classifier_guidance"]:
        classifier = hydra.utils.instantiate(cfg.classifier)
        classifier.load_state_dict(
            torch.load(
                os.path.join(cfg.classifier_path, "model", "classifier.pt"),
                map_location=cfg.device,
            )
        )
        workspace.env.cond_mask_prob = 0.1

        workspace.agent.model = ClassifierGuidedSampleModel(workspace.agent.model, classifier, 1)
        results_dict = workspace.env.simulate(
            workspace, real_time=False, lambda_values=cfg.lambda_values
        )

        rewards = [v for k, v in results_dict.items() if k.endswith("/reward_mean")]
        rewards_std = [v for k, v in results_dict.items() if k.endswith("/reward_std")]
        terminals = [
            v for k, v in results_dict.items() if k.endswith("/terminals_mean")
        ]

        print(rewards)
        print(rewards_std)
        print(terminals)
        plt.bar(range(len(rewards)), rewards)
        plt.xticks(range(len(cfg.lambda_values)), cfg.lambda_values)
        plt.xlabel("Lambda")
        plt.ylabel("Velocity tracking return")
        plt.show()


if __name__ == "__main__":
    main()
