import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import torch

class FootGrid():
    def __init__(self, grid_limits, num_bins, device):
        self.grid_limits = grid_limits
        self.num_bins = num_bins
        self.device = device

        grids = []
        for i in range(4):
            # (num_bins, num_bins, (anchor, l, w))
            grid = torch.zeros((num_bins, num_bins, 4)).to(device)

            x_min, x_max, y_min, y_max = self.grid_limits[i]
            x_bins = torch.linspace(x_min, x_max, num_bins + 1)
            y_bins = torch.linspace(y_min, y_max, num_bins + 1)
            l = x_bins[1] - x_bins[0]
            w = y_bins[1] - y_bins[0]

            anchors = torch.meshgrid(x_bins[:-1], y_bins[:-1])
            data = [anchors[0], anchors[1], l, w]
            for i in range(4):
                grid[:, :, i] = data[i]
            grids.append(grid.reshape(1, -1, 4))
        self.grid = torch.cat(grids, dim=0)
    
    def get_active_grids(self, state):
        """sets the grid index to 1 if the foot is in the grid limits for any timestep"""
        foot_pos = state[:, :, 47:].reshape(*state.shape[:2], 4, 3).unsqueeze(3)
        grids = self.grid.unsqueeze(0).unsqueeze(0)

        min_xy = grids[..., :2]
        max_xy = min_xy + grids[..., 2:]

        # Check if the foot position falls within the grid squares and below a z-coordinate of -0.57
        active = ((foot_pos[..., 0] >= min_xy[..., 0]) & (foot_pos[..., 0] <= max_xy[..., 0]) &
                  (foot_pos[..., 1] >= min_xy[..., 1]) & (foot_pos[..., 1] <= max_xy[..., 1]) &
                  (foot_pos[..., 2] < -0.57))

        # (B, num_feet, num_bins**2)
        return active.int().any(dim=1)
            
    
    def get_avoid_grids(self, state):
        """Sets the grid index to 1 if the grid is over the gap"""
        gap_x, gap_delta = 1.4, 0.2

        com = state[:, 36:38].unsqueeze(1).unsqueeze(1)
        orientation = state[:, 38:47].reshape(-1, 3, 3)[:, :2, :2]
        orientation = orientation.unsqueeze(1).unsqueeze(1)
        grid = self.grid.unsqueeze(0)

        min_xy = torch.einsum('abci,abcij->abcj', (grid[..., :2], orientation)) + com
        max_xy = torch.einsum('abci,abcij->abcj', (grid[..., :2] + grid[..., 2:], orientation)) + com
        active = (((min_xy[..., 0] >= gap_x) & (min_xy[..., 0] <= gap_x + gap_delta)) | 
                  ((max_xy[..., 0] >= gap_x) & (max_xy[..., 0] <= gap_x + gap_delta)))

        return active.reshape(state.shape[0], 1, -1).to(torch.float32)


def test():
    foot_grid = FootGrid(
        torch.tensor([[0.22, 0.48, 0, 0.24], 
                  [0.22, 0.48, -0.24, 0],
                  [-0.45, -0.2, 0, 0.24],
                  [-0.45, -0.2, -0.24, 0]]), 4, 'cpu')
    state = torch.tensor([[-1.68466996e-02, -3.04289125e-02,  9.99394894e-01,
        -4.93928820e-01,  6.91931367e-01, -1.60921502e+00,
         2.44295478e-01,  5.03715038e-01, -6.90906048e-01,
        -2.27615863e-01, -6.44318700e-01,  6.14952326e-01,
         2.95213401e-01, -7.99368382e-01,  1.66800761e+00,
        -2.33724713e-02, -1.66417092e-01, -5.33219039e-01,
        -8.90503265e-03, -1.06666911e+00,  1.92639649e-01,
         9.23587978e-01, -2.24852376e-02,  7.74194121e-01,
         4.39699888e-01,  8.02999914e-01, -1.25749528e-01,
         3.20933312e-02, -3.01745701e+00, -1.44256604e+00,
         2.25706518e-01, -3.73259664e-01,  1.02087870e-01,
         2.32830644e-10,  5.82076609e-11, -1.16415322e-10,
         2.09192491e+00,  2.82399297e+00,  6.06466889e-01,
         7.11758196e-01,  7.02222466e-01, -1.68467853e-02,
        -7.02361822e-01,  7.11169422e-01, -3.04289851e-02,
        -9.38699953e-03,  3.34906206e-02,  9.99394953e-01,
         5.06297052e-01,  1.12631120e-01, -4.11022872e-01,
         3.66782844e-01, -1.47123724e-01, -5.99356174e-01,
        -2.74984330e-01,  1.72528744e-01, -5.98945141e-01,
        -4.42486733e-01, -1.86357990e-01, -4.11051005e-01]])
    state = state.repeat(2, 1)
    active_grids = foot_grid.get_active_grids(state)

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        state[:, 36] = frame / 10  # vary from 2 to 4
        avoid_grids, max_xys, min_xys = foot_grid.get_avoid_grids(state)

        for i in range(4):
            avoid = avoid_grids[0, i]
            max_xy = max_xys[0, i]
            ax.scatter(max_xy[avoid == 1, 0], max_xy[avoid == 1, 1], marker='x')
            ax.scatter(max_xy[avoid == 0, 0], max_xy[avoid == 0, 1], marker='o')

    ani = FuncAnimation(fig, update, frames=range(20, 40), repeat=False)
    plt.show()


if __name__ == "__main__":
    test()