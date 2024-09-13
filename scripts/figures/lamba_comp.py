import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})

rewards = [
    0.55315906,
    0.8154814,
    0.8318411,
    0.8425057,
    0.8408836,
    0.77744186,
    0.68283486,
]
stds = [
    0.13046372,
    0.12729926,
    0.12538742,
    0.11912741,
    0.12043013,
    0.1482092,
    0.16127522,
]
terminals = [0.06, 0.27, 0.28, 0.23, 0.31, 2.08, 7.43]
x_labels = [0, 1, 1.2, 1.5, 2, 5, 10]

fig, ax1 = plt.subplots(figsize=(14, 10))
fig.subplots_adjust(right=0.88)  # Make room for the second y-axis label

x = np.arange(len(rewards))

# Plot rewards on the primary y-axis
color1 = "#1f77b4"  # A professional blue color
ax1.bar(
    x - 0.2,
    rewards,
    yerr=stds,
    width=0.35,
    color=color1,
    capsize=5,
    error_kw=dict(ecolor="black", lw=1, capsize=3, capthick=1),
    label="Rewards",
)
ax1.set_xlabel(r"$\lambda$", fontsize=24, labelpad=20)
ax1.set_ylabel("Rewards", fontsize=24, labelpad=20)
ax1.tick_params(axis="y", labelcolor=color1)

# Create a secondary y-axis for terminals
ax2 = ax1.twinx()
color2 = "#ff7f0e"  # A complementary orange color
ax2.bar(x + 0.2, terminals, width=0.35, color=color2, label="Terminations")
ax2.set_ylabel("Terminations", fontsize=24, labelpad=20)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.grid(False)

# Set x-axis labels and title
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_title(
    "Velocity Tracking Reward and Terminations for Different λ Values",
    fontsize=24,
    pad=20,
)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    frameon=True,
    framealpha=0.7,
    fontsize=24,
    fancybox=True,
    facecolor="white",
)

# Add grid for better readability
ax1.grid(True, linestyle="--")
ax1.set_axisbelow(True)

# Ensure y-axes have a reasonable number of ticks
ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Add a box around the plot
for spine in ax1.spines.values():
    spine.set_visible(True)
ax1.patch.set_edgecolor("black")
ax1.patch.set_linewidth(1.5)

# Tight layout and save with high DPI
plt.tight_layout()
plt.savefig("lambda_comp.pdf", format="pdf", dpi=300, bbox_inches="tight")
