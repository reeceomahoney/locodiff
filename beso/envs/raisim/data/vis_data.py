import numpy as np
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
import hydra


obs = np.load('beso/envs/raisim/data/random_acts.npy', allow_pickle=True).item()['observations']
actions = np.load('beso/envs/raisim/data/random_acts.npy', allow_pickle=True).item()['actions']

actions = actions.reshape(-1, 12)
norms = np.linalg.norm(actions, axis=-1)

joint_velocities = obs[..., 18:30]
joint_velocities = joint_velocities.reshape(-1, 12)
norms = np.linalg.norm(joint_velocities, axis=-1)
print(np.quantile(norms, 0.1))

orientation = obs[..., :3]
gravity_vector = np.array([0, 0, 1])
dot_product = np.dot(orientation, gravity_vector)
angles = np.arccos(dot_product)
print(np.quantile(angles, 0.1))


# ang_vel = obs[..., 15:18]
# norms = ang_vel.norm(dim=-1).cpu().numpy()

# Plot the distribution of the norms
# plt.hist(norms, bins=50)
# plt.xlabel('Norm')
# plt.ylabel('Frequency')
# plt.title('Distribution of Joint Velocity Norms')
# plt.show()
