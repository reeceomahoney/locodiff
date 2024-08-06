import numpy as np
import matplotlib.pyplot as plt

target_data = np.load("scripts/fig/target.npy")
constraint_data = np.load("scripts/fig/constraint.npy")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

lambda_values = [-5, -2, 0, 1, 2, 5, 10]
x_coords = range(1, len(lambda_values) + 1)

box_color = dict(facecolor='lightblue')
median_color = dict(color='red')

ax[0].boxplot(target_data.T, patch_artist=True, showfliers=False, boxprops=box_color, medianprops=median_color)
ax[0].set_title("Velocity Target")
ax[0].set_xticks(x_coords)
ax[0].set_xticklabels(lambda_values)
ax[0].set_xlabel("Lambda")
ax[0].set_ylabel("Mean Episodic Reward")

ax[1].boxplot(constraint_data.T, patch_artist=True, showfliers=False, boxprops=box_color, medianprops=median_color)
ax[1].set_title("Velocity Constraint")
ax[1].set_xticks(x_coords)
ax[1].set_xticklabels(lambda_values)
ax[1].set_xlabel("Lambda")
ax[1].set_ylabel("Mean Episodic Reward")



plt.tight_layout()
plt.show()
