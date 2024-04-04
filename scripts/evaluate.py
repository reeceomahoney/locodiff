import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


@hydra.main(
    config_path="../configs", config_name="evaluate_raisim.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    if cfg.log_wandb:
        wandb.init(project="beso_eval", mode="disabled", config=wandb.config)

    # config
    cfg_store_path = os.path.join(
        os.getcwd(), cfg.model_store_path, ".hydra/config.yaml"
    )
    model_cfg = OmegaConf.load(cfg_store_path)
    model_cfg.device = cfg.device
    model_cfg.workspaces["device"] = cfg["device"]
    model_cfg.agents["device"] = cfg["device"]
    model_cfg.env["num_envs"] = 1
    model_cfg.env["server_port"] = 8081
    model_cfg.env["max_time"] = 6

    # set the observation dimension
    model_cfg["obs_dim"] = 34
    model_cfg["pred_obs_dim"] = 34

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    workspace_manager = hydra.utils.instantiate(model_cfg.workspaces)
    agent = hydra.utils.instantiate(model_cfg.agents)

    # get the scaler instance and set the bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.load_pretrained_model(cfg.model_store_path)

    # set new noise limits
    agent.sigma_max = cfg.sigma_max
    agent.sigma_min = cfg.sigma_min

    # initialize the environment
    workspace_manager.init_env(model_cfg, model_cfg.data_path)

    # set seeds
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Evaluate
    if cfg["test_rollout"]:
        workspace_manager.eval_n_times = cfg["num_runs"]
        results_dict = workspace_manager.test_agent(
            agent, n_inference_steps=cfg["n_inference_steps"], real_time=True
        )
        print(results_dict)
    else:
        dataloader = workspace_manager.make_dataloaders()["test"]
        batch = next(iter(dataloader))
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        if cfg["test_diffusion_steps"]:
            inference_steps = [1, 2, 3, 4, 5, 10, 20, 40, 50]
            results = []
            for step in inference_steps:
                agent.num_sampling_steps = step
                info = agent.evaluate(batch)
                results.append(info["timestep_mse"].cpu().numpy())

            for i, result in enumerate(results):
                plt.plot(
                    np.arange(1, len(result) + 1),
                    result,
                    label=f"{inference_steps[i]} inference steps",
                )
        elif cfg["test_observation_error"]:
            info = agent.evaluate(batch)
            results = info["mse"].cpu().numpy().mean(axis=(0, 1))
            plt.bar(range(len(results)), results)

        plt.yscale("log")
        plt.legend()
        plt.savefig("test.png")

if __name__ == "__main__":
    main()
