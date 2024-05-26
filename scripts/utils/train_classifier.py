import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb


def expand(x, B):
    return x.unsqueeze(0).expand(B, -1, -1).clone()


def calculate_value_function(reward, prev_value, gamma):
    reward[:, -1] = 0
    prev_value[:, :-1] = prev_value[:, 1:]
    value = reward + gamma * prev_value
    return value


@hydra.main(config_path="../configs", config_name="classifier.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    # config
    cfg_store_path = os.path.join(
        os.getcwd(), cfg.model_store_path, ".hydra/config.yaml"
    )
    model_cfg = OmegaConf.load(cfg_store_path)

    # init wandb
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    mode = "online" if sys.gettrace() is None else "disabled"
    wandb.init(
        project=cfg.wandb.project, mode=mode, config=wandb.config, dir=output_dir
    )

    # set seeds
    np.random.seed(model_cfg.seed)
    torch.manual_seed(model_cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    agent = hydra.utils.instantiate(model_cfg.agents)
    agent.load_pretrained_model(cfg.model_store_path)
    agent.num_sampling_steps = cfg.inference_steps

    classifier = hydra.utils.instantiate(cfg.classifier)
    optimizer = hydra.utils.instantiate(cfg.optimizer, classifier.parameters())
    loss = torch.nn.MSELoss()

    generator = iter(agent.train_loader)
    test_generator = iter(agent.test_loader)

    # Train
    for step in tqdm(range(int(cfg.train_steps))):
        try:
            batch = next(generator)
        except StopIteration:
            generator = iter(agent.train_loader)
            batch = next(generator)

        # Generate random number of inference steps
        num_steps = np.random.randint(1, cfg.inference_steps + 1)

        info = agent.evaluate(batch, num_steps=num_steps)
        epsilon, cond, goal = info["inputs"]

        T_cond = 2
        value = batch["value"][:, T_cond - 1:].to("cuda")
        value = classifier.normalize(value)

        pred_value = classifier(epsilon, cond, goal)
        output = loss(pred_value, value)

        optimizer.zero_grad()
        output.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=0.5)
        optimizer.step()

        # Evaluate test loss
        if not step % cfg.test_interval:
            with torch.no_grad():
                try:
                    test_batch = next(test_generator)
                except StopIteration:
                    test_generator = iter(agent.test_loader)
                    test_batch = next(test_generator)

                test_info = agent.evaluate(test_batch, num_steps=num_steps)
                test_epsilon, test_cond, test_goal = test_info["inputs"]

                test_value = test_batch["value"][:, T_cond - 1:].to("cuda")
                test_value = classifier.normalize(test_value)

                test_pred_value = classifier(test_epsilon, test_cond, test_goal)
                test_output = loss(test_pred_value, test_value)

                wandb.log({"test_loss": test_output.item()}, step=step)

        if not step % cfg.log_interval:
            wandb.log({"loss": output.item()}, step=step)

        if not step % cfg.save_interval:
            torch.save(
                classifier.state_dict(), os.path.join(output_dir, "classifier.pth")
            )
            print("Model saved!")


if __name__ == "__main__":
    main()
