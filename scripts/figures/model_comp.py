import numpy as np

import matplotlib.pyplot as plt

# Set the style
plt.style.use("seaborn-v0_8")

data = [[0.91, 0.92, 0.92, 0.88], [0.88, 0.80, 0, 0], [0.69, 0, 0, 0]]
x_labels = ["Forwards", "Backwards", "Lateral", "Rotation"]  # Text labels for x-axis

# Set the width of each bar
bar_width = 0.2

# Plot the data as side-by-side bar chart
x = np.arange(len(data[0]))  # Generate x-axis values
plt.bar(x - bar_width, data[0], width=bar_width, label="Ground Truth")
plt.bar(x, data[1], width=bar_width, label="SDE (Ours)")
plt.bar(x + bar_width, data[2], width=bar_width, label="DDPM")

# Replace x-axis ticks with text labels
plt.xticks(x, x_labels)

# Add labels and title
plt.xlabel("Commands")
plt.ylabel("Rewards")
plt.title("Velocity Tracking Reward for Different Commands")

# Add legend with a box
plt.legend(frameon=True, fancybox=True, borderpad=1, facecolor="white")

# Show the plot
plt.show()
