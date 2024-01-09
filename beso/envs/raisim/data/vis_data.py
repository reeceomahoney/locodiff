import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
import hydra


cfg_store_path = 'logs/raisim/runs/2024-01-09/12-01-22/.hydra/config.yaml'
model_cfg = OmegaConf.load(cfg_store_path) 
workspace_manager = hydra.utils.instantiate(model_cfg.workspaces)

obs = workspace_manager.train_set.dataset.dataset.get_all_observations()
obs = workspace_manager.scaler.scale_input(obs)
joint_velocities = obs[..., 18:30]
print(joint_velocities.min(), joint_velocities.max())

# Calculate the norms of the joint velocities
norms = joint_velocities.norm(dim=-1).cpu().numpy()

# Plot the distribution of the norms
plt.hist(norms, bins=50)
plt.xlabel('Norm')
plt.ylabel('Frequency')
plt.title('Distribution of Joint Velocity Norms')
plt.show()
