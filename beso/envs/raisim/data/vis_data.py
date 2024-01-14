import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
import hydra


cfg_store_path = 'logs/raisim/runs/2024-01-14/13-33-11/.hydra/config.yaml'
model_cfg = OmegaConf.load(cfg_store_path) 
workspace_manager = hydra.utils.instantiate(model_cfg.workspaces)

obs = workspace_manager.train_set.dataset.dataset.get_all_observations()
obs = workspace_manager.scaler.scale_input(obs)

joint_velocities = obs[..., 18:30]
norms = joint_velocities.norm(dim=-1).cpu().numpy()
print(norms.mean())

# orientation = obs[..., :3]
# norms = orientation.norm(dim=-1).cpu().numpy()

# ang_vel = obs[..., 15:18]
# norms = ang_vel.norm(dim=-1).cpu().numpy()

# Plot the distribution of the norms
plt.hist(norms, bins=50)
plt.xlabel('Norm')
plt.ylabel('Frequency')
plt.title('Distribution of Joint Velocity Norms')
plt.show()
