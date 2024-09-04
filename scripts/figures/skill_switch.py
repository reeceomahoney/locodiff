import numpy as np

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})

data = np.load("scripts/figures/skill_switch.npy", allow_pickle=True).item()
reward = data["reward"]
height = data["height"]
x = np.arange(len(reward)) / 25

# Calculate running mean and variance
window_size = 20
reward_mean = np.convolve(reward, np.ones(window_size) / window_size, mode="valid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.patch.set_edgecolor("black")
ax1.patch.set_linewidth(1.5)
ax2.patch.set_edgecolor("black")
ax2.patch.set_linewidth(1.5)

ax1.plot(x[window_size - 1 :], reward_mean, label="Reward (Running Mean)", color="blue")
ax1.plot(x, reward, label="Reward", alpha=0.3)
ax1.set_xlabel("Time (s)", fontsize=24, labelpad=20)
ax1.set_ylabel("Rewards", fontsize=24, labelpad=20)
ax1.tick_params(labelbottom=True)

ax2.plot(x, height, label="Height", color="red")
ax2.set_xlabel("Time (s)", fontsize=24, labelpad=20)
ax2.set_ylabel("Height", fontsize=24, labelpad=20)

ax1.set_title("Velocity Tracking Reward during skill change", fontsize=24, pad=20)
legend = ax1.legend(frameon=True, fancybox=True, borderpad=0.5, facecolor="white", fontsize=24)
legend.get_frame().set_linewidth(1.5)


plt.show()
