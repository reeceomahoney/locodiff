import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})

data = np.load("switch_data.npy", allow_pickle=True).item()
reward = data["reward"]
height = data["height"]
timestamp = data["timestamp"]  # Use the timestamp from the data

# Calculate running mean
window_size = 200
reward_mean = np.convolve(reward, np.ones(window_size) / window_size, mode="valid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(36, 8), sharex=True)
ax1.patch.set_edgecolor("black")
ax1.patch.set_linewidth(1.5)
ax2.patch.set_edgecolor("black")
ax2.patch.set_linewidth(1.5)

ax1.plot(
    timestamp[window_size - 1 :],
    reward_mean,
    label="Reward (Running Mean)",
    color="blue",
)
ax1.plot(timestamp, reward, label="Reward", alpha=0.3)
ax1.set_xlabel("Time (s)", fontsize=24, labelpad=20)
ax1.set_ylabel("Rewards", fontsize=24, labelpad=20)
ax1.tick_params(labelbottom=True)

ax2.plot(timestamp, height, label="Height", color="red")
ax2.set_xlabel("Time (s)", fontsize=24, labelpad=5)
ax2.set_ylabel("Height (m)", fontsize=24, labelpad=20)

ax1.set_title("Velocity tracking reward during skill change", fontsize=24, pad=20)

# Add vertical line at timestamp = 5 to both axes
ax1.axvline(x=17.5, color="green", linestyle="--", linewidth=2, label="Skill Change")
ax2.axvline(x=17.5, color="green", linestyle="--", linewidth=2)

legend = ax1.legend(
    frameon=True, fancybox=True, borderpad=0.5, facecolor="white", fontsize=24
)
legend.get_frame().set_linewidth(1.5)

plt.show()
