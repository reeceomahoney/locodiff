import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import socket

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from env.raisim_env import RaisimEnv


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


@hydra.main(config_path="../configs", config_name="evaluate.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.log_wandb:
        wandb.init(project="beso_eval", mode="disabled", config=wandb.config)

    # config
    cfg_store_path = os.path.join(
        os.getcwd(), cfg.model_store_path, ".hydra/config.yaml"
    )
    model_cfg = OmegaConf.load(cfg_store_path)
    model_cfg.device = cfg.device
    model_cfg.agents["device"] = cfg["device"]
    model_cfg.env["num_envs"] = 1
    model_cfg.env["server_port"] = 8081
    model_cfg.env["max_time"] = 6

    # set the observation dimension
    model_cfg["obs_dim"] = 35
    model_cfg["pred_obs_dim"] = 35
    model_cfg["T_action"] = 1

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    agent = hydra.utils.instantiate(model_cfg.agents)
    agent.load_pretrained_model(cfg.model_store_path)
    env = RaisimEnv(model_cfg)

    # set new noise limits
    agent.sigma_max = cfg.sigma_max
    agent.sigma_min = cfg.sigma_min

    # set seeds
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Evaluate
    if cfg["test_rollout"]:
        env.eval_n_times = cfg["num_runs"]
        results_dict = env.simulate(
            agent, n_inference_steps=cfg["n_inference_steps"], real_time=True
        )
        print(results_dict)
    else:
        dataloader = agent.test_loader
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        if cfg["test_timestep_mse"]:
            inference_steps = [1, 2, 3, 4, 5, 10, 20, 40, 50]
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
            inference_steps = [1, 2, 3, 4, 5, 10, 20, 40, 50]
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
            agent.num_sampling_steps = 50
            info = agent.evaluate(batch)
            obs = batch["observation"].cpu().numpy()
            vel_cmd = batch["cmd"].cpu().numpy()
            results = info["prediction"].cpu().numpy()

            fig, axs = plt.subplots(4, 4, figsize=(15, 15))
            axs = axs.flatten()

            T_cond = model_cfg["T_cond"]
            for i in range(16):
                gt = obs[i, T_cond - 1 :, :2]
                axs[i].plot(gt[:, 0], gt[:, 1], "o-", label="observed")
                axs[i].plot(gt[0, 0], gt[0, 1], "go", label="Start")
                axs[i].plot(gt[-1, 0], gt[-1, 1], "ro", label="End")

                pred = results[i, :, :2]
                axs[i].plot(pred[:, 0], pred[:, 1], "x-", label="predicted")
                axs[i].plot(pred[0, 0], pred[0, 1], "gx", label="Start")
                axs[i].plot(pred[-1, 0], pred[-1, 1], "rx", label="End")

                vel_cmd_str = ", ".join(f"{num:.2f}" for num in vel_cmd[i])
                axs[i].set_title(f"cmd: [ {vel_cmd_str} ]")
                axs[i].legend()

            plt.tight_layout()
            plt.show()

        if not cfg["visualize x-y trajectory"]:
            plt.legend()
            if socket.gethostname() == "ori-drs-sid":
                plt.show()
            else:
                plt.savefig("results.png")

if __name__ == "__main__":
    main()
