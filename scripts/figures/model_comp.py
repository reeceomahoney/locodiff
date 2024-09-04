import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})
bar_width = 0.2

data = [[0.91, 0.92, 0.92, 0.88], [0.88, 0.81, 0, 0], [0.69, 0.64, 0, 0]]
x = np.arange(len(data[0]))
x_labels = ["Forwards", "Backwards", "Lateral", "Rotation"]  # Text labels for x-axis

fig, ax = plt.subplots(figsize=(14, 10))
ax.patch.set_edgecolor("black")
ax.patch.set_linewidth(1.5)

ax.bar(x - bar_width, data[0], width=bar_width, label="Ground Truth")
ax.bar(x, data[1], width=bar_width, label="SDE (Ours)")
ax.bar(x + bar_width, data[2], width=bar_width, label="DDPM")

ax.set_xticks(x, x_labels)

ax.set_xlabel("Commands", fontsize=24, labelpad=20)
ax.set_ylabel("Rewards", fontsize=24, labelpad=20)
ax.set_title("Velocity Tracking Reward for Different Commands", fontsize=24, pad=20)

ax.legend(frameon=True, fancybox=True, borderpad=1, facecolor="white")
plt.show()
