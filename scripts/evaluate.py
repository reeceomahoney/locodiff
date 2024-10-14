import logging
import os

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from locodiff.classifier import ClassifierGuidedSampleModel
from locodiff.samplers import get_sampler

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
    model_cfg.env.eval_times = cfg.num_runs
    model_cfg.T_action = 1
    model_cfg.use_ema = False
    model_cfg.num_envs = 1 * len(cfg.lambda_values)
    model_cfg.env.impl.num_envs = 1 * len(cfg.lambda_values)
    model_cfg.sampling_steps = cfg.sampling_steps
    model_cfg.wandb_mode = "disabled"

    # set seeds
    np.random.seed(model_cfg.env.seed)
    torch.manual_seed(model_cfg.env.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    workspace = hydra.utils.instantiate(model_cfg)
    workspace.load(cfg.model_store_path)
    # agent = torch.jit.load(f"data/models/policy_{cfg.device}.pt")

    # Evaluate
    if cfg["test_rollout"]:
        results_dict = workspace.env.simulate(
            workspace, real_time=True, lambda_values=[0]
        )
        print(results_dict)
    if cfg["test_reward_lambda"]:
        results_dict = workspace.env.simulate(
            workspace, real_time=False, lambda_values=cfg.lambda_values
        )
        rewards = [v for k, v in results_dict.items() if k.endswith("/reward_mean")]
        rewards_std = [v for k, v in results_dict.items() if k.endswith("/reward_std")]
        terminals = [
            v for k, v in results_dict.items() if k.endswith("/terminals_mean")
        ]

        print(rewards)
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

        workspace.agent.model = ClassifierGuidedSampleModel(
            workspace.agent.model, classifier, 1
        )
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
        # plt.bar(range(len(rewards)), rewards)
        # plt.xticks(range(len(cfg.lambda_values)), cfg.lambda_values)
        # plt.xlabel("Lambda")
        # plt.ylabel("Velocity tracking return")
        # plt.show()
    if cfg["test_encoder"]:
        workspace.agent.sampler = get_sampler("encoder_ddim")
        batch = next(iter(workspace.test_loader))
        encoded_batch = workspace.encode(batch)
        stds = encoded_batch.std(dim=(0,1))
        stds = stds.cpu().numpy()
        plt.bar(range(len(stds)), stds)
        # plt.show()
        plt.savefig("foo.png")


if __name__ == "__main__":
    main()
