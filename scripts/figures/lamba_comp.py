import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})

data = [0.5145297, 0.85218513, 0.8606936, 0.8706338, 0.8789472, 0.81202245, 0.51021385]
x_labels = [0, 1, 1.2, 1.5, 2, 5, 10]
fig, ax = plt.subplots(figsize=(14, 10))
ax.patch.set_edgecolor("black")
ax.patch.set_linewidth(1.5)

x = np.arange(len(data))
ax.bar(x, data)
ax.set_xticks(x, x_labels)

ax.set_xlabel("Lambda", fontsize=24, labelpad=20)
ax.set_ylabel("Rewards", fontsize=24, labelpad=20)
ax.set_title(
    "Velocity Tracking Reward for Different Lambda Values", fontsize=24, pad=20
)

plt.show()
