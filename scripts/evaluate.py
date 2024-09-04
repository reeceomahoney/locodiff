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
        lambda_values = [0, 1, 2, 5, 10, 20, 50, 100, 150]
        # lambda_values = [0, 1, 1.2, 1.5, 1.7, 2, 5, 10, 20]

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
    else:
        dataloader = agent.test_loader
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        if cfg["test_timestep_mse"]:
            inference_steps = [1, 2, 3, 4, 5, 10, 20, 50]
            results = []
            for step in tqdm(inference_steps):
                agent.num_sampling_steps = step
                info = agent.evaluate(batch)
                results.append(info["timestep_mse"].cpu().numpy())

            for i, result in enumerate(results):
                plt.plot(
                    np.arange(1, len(result) + 1),
                    result,
                    label=f"{inference_steps[i]} inference steps",
                )
        if cfg["test_total_mse"]:
            inference_steps = [1, 2, 3, 4, 5, 10, 20, 50]
            results = []
            for step in inference_steps:
                agent.num_sampling_steps = step
                info = agent.evaluate(batch)
                results.append(info["total_mse"])
            plt.plot(inference_steps, results, "x")
        if cfg["test_observation_error"]:
            info = agent.evaluate(batch)
            results = info["mse"].cpu().numpy().mean(axis=(0, 1))
            plt.bar(range(len(results)), results)
        if cfg["visualize x-y trajectory"]:
            agent.num_sampling_steps = 10
            T_cond = model_cfg["T_cond"]
            # batch = {k: v[:16] for k, v in batch.items()}

            obs = batch["observation"].cpu().numpy()
            obs[:, :, :2] -= obs[:, T_cond - 1 : T_cond, :2]

            info = agent.evaluate(batch)
            goal = info["goal"].cpu().numpy()
            results = info["prediction"].cpu().numpy()

            fig, axs = plt.subplots(4, 4, figsize=(15, 15))
            axs = axs.flatten()

            T_cond = model_cfg["T_cond"]
            for i in range(16):
                gt = obs[i, T_cond - 1 :, :2]
                axs[i].plot(gt[:, 0], gt[:, 1], "o-", label="observed")
                axs[i].plot(goal[i, 0], goal[i, 1], "rx", label="Goal")

                gt_vel = obs[i, T_cond - 1 :, 33:36]
                gt_ori = obs[i, T_cond - 1 :, 2:6]
                gt_ori = np.roll(gt_ori, shift=-1, axis=1)
                rot = R.from_quat(gt_ori).as_matrix()
                gt_vel = np.einsum("bij, bj -> bi", rot, gt_vel)
                for j in range(0, len(gt_vel), 10):
                    axs[i].quiver(
                        gt[j, 0],
                        gt[j, 1],
                        gt_vel[j, 0],
                        gt_vel[j, 1],
                        scale=5,
                        width=0.005,
                        color="r",
                    )

                pred = results[i, :, :2]
                axs[i].plot(pred[:, 0], pred[:, 1], "x-", label="predicted")

                axs[i].legend()

            plt.tight_layout()
            plt.savefig("results.png")


if __name__ == "__main__":
    main()
