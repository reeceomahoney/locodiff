import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})
bar_width = 0.2

data = [[0.91, 0.92, 0.92, 0.88], [0.88, 0.81, 0.77, 0.84], [0.69, 0.67, 0.68, 0]]
x = np.arange(len(data[0]))
x_labels = ["Forwards", "Backwards", "Lateral", "Rotation"]  # Text labels for x-axis

fig, ax = plt.subplots(figsize=(14, 11))
ax.patch.set_edgecolor("black")
ax.patch.set_linewidth(1.5)

ax.bar(x - bar_width, data[0], width=bar_width, label="Ground Truth")
ax.bar(x, data[1], width=bar_width, label="SDE (Ours)")
ax.bar(x + bar_width, data[2], width=bar_width, label="DDPM")

ax.set_xticks(x, x_labels)

ax.set_xlabel("Commands", fontsize=24, labelpad=20)
ax.set_ylabel("Rewards", fontsize=24, labelpad=20)
ax.set_title("Velocity Tracking Reward for Different Commands", fontsize=24, pad=20)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.85])

legend = ax.legend(
    frameon=True,
    fancybox=True,
    borderpad=0.5,
    facecolor="white",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=24,
)
legend.get_frame().set_linewidth(1.5)

plt.show()
