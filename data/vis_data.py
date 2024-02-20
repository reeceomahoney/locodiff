import numpy as np
import matplotlib.pyplot as plt


dataset = "rand_feet_com"
obs = np.load("beso/envs/raisim/data/" + dataset + ".npy", allow_pickle=True).item()[
    "observations"
]

feet_pos = obs[..., -12:].reshape(-1, 4, 3)
feet_pos_x = feet_pos[..., 0]
feet_pos_y = feet_pos[..., 1]
feet_pos_z = feet_pos[..., 2]

mask = feet_pos_z < -0.57
feet_pos_x = feet_pos_x[mask]
feet_pos_y = feet_pos_y[mask]

grid = np.array([
    [0.05, 0.61, 0, 0.36],
    [0.05, 0.61, -0.36, 0],
    [-0.61, -0.05, 0, 0.36],
    [-0.61, -0.05, -0.36, 0],
])
corners = []
for min_x, max_x, min_y, max_y in grid:
    # Calculate the corners
    corners.append([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, min_y],
        [max_x, max_y],
    ])
corners = np.array(corners)


plt.plot(feet_pos_x, feet_pos_y, "o")
plt.plot(corners[..., 0], corners[..., 1], "o")
plt.show()
