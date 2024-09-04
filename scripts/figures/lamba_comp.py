import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})

data = [0.5455793, 0.67960495, 0.8092913, 0.8293958, 0.86788064, 0.87723356]
x_labels = [0, 1, 1.5, 2, 3, 5]

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
