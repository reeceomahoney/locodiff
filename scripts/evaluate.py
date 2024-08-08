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
    model_cfg.agents["device"] = cfg["device"]
    model_cfg.env["num_envs"] = 50
    model_cfg.env["server_port"] = 8081
    model_cfg.env["max_time"] = 100
    model_cfg["T_action"] = 1
    model_cfg["use_ema"] = False

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
        results_dict = env.simulate(
            agent, n_inference_steps=cfg["n_inference_steps"], real_time=True
        )
        print(results_dict)
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
        if cfg["test_cond_lambda"]:
            agent.num_sampling_steps = 10
            lambdas = [0, 1e-6, 2e-3]
            batch = {k: v[3:4] for k, v in batch.items()}

            fix, axs = plt.subplots(2, 3, figsize=(15, 10))
            axs = axs.flatten()

            obs = batch["observation"].cpu().numpy()
            obs[:, :, :2] -= obs[:, model_cfg["T_cond"] - 1, :2]

            for i, l in enumerate(lambdas):
                info = agent.evaluate(batch, cond_lambda=l)
                goal = info["goal"].cpu().numpy()
                pred = info["prediction"].cpu().numpy()[0, :, :2]
                gt = obs[0, model_cfg["T_cond"] - 1 :, :2]

                axs[i].title.set_text(f"Lambda: {l}")
                axs[i].plot(gt[:, 0], gt[:, 1], "o-", label="observed")
                axs[i].plot(pred[:, 0], pred[:, 1], "x-", label="predicted")
                axs[i].plot(goal[0, 0], goal[0, 1], "rx", label="Goal")

                axs[i].legend()
        if cfg["test_t_sne"]:
            env.eval_n_times = 10
            results_dict = env.simulate(
                agent, n_inference_steps=cfg["n_inference_steps"], real_time=True
            )

            obs_dim = 36
            obs = results_dict["all_obs"]
            vels = np.concatenate([obs[..., 33:35], obs[..., 20:21]], axis=-1)
            cmds = results_dict["all_cmds"]

            error = (vels - cmds) ** 2
            error = error.mean(axis=(0, 1))
            print(error)

            # dataset = np.load("data/datasets/walk.npy", allow_pickle=True).item()
            # dataset_obs = dataset["observations"].reshape(-1, obs_dim)[:12450, 6:18]

            # tsne_1 = TSNE(n_components=2).fit_transform(obs)
            # tsne_2 = TSNE(n_components=2).fit_transform(dataset_obs)

            # plt.scatter(tsne_1[:, 0], tsne_1[:, 1], label="Ground truth")
            # plt.scatter(tsne_2[:, 0], tsne_2[:, 1], label="Generated")
            # plt.legend()
            # plt.show()

        # if not cfg["visualize x-y trajectory"]:
        #     plt.legend()
        #     if socket.gethostname() == "ori-drs-sid":gg
        if cfg["test_reward_lambda"]:
            env.eval_n_times = 1
            env.eval_n_steps = 250
            lambda_values = [-5, -2, 0, 1, 2, 5, 10]
            rewards = []

            for i, lam in enumerate(lambda_values):
                agent.cond_lambda = lam
                results_dict = env.simulate(
                    agent, n_inference_steps=cfg["n_inference_steps"], real_time=True
                )
                reward = np.exp(-((results_dict["x_pos"] - 0.6) ** 2)*10)
                # reward = ((results_dict["x_pos"] > 0.5 ) & (results_dict["x_pos"] < 0.8)).astype(float)
                rewards.append(reward.mean(axis=-1))

            x_coords = range(1, len(lambda_values) + 1)
            rewards = np.array(rewards)
            np.save("target.npy", rewards)

            box = plt.boxplot(rewards.T, patch_artist=True, showfliers=False)

            for patch in box["boxes"]:
                patch.set_facecolor("lightblue")

            plt.xticks(x_coords, lambda_values)
            plt.xlabel('Lambda values')
            plt.ylabel('Rewards')
            plt.show()

        # if not cfg["visualize x-y trajectory"]:
        #     plt.legend()
        #     if socket.gethostname() == "ori-drs-sid":
        #         plt.show()
        #     else:
        #         plt.savefig("results.png")


if __name__ == "__main__":
    main()
