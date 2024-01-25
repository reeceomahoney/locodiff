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
            grids.append(grid.reshape(-1, 4))
        self.grid = {key: grid for key, grid in zip(['FL', 'FR', 'HL', 'HR'], grids)}
    
    def get_active_grids(self, state):
        """sets the grid index to 1 if the foot is in the grid"""
        # NB: not every foot in the dataset falls within the a grid. May have to change this.
        foot_pos = state[:, None, 48:].reshape(-1, 4, 3)
        grids = torch.stack(list(self.grid.values()))
        grids = grids.unsqueeze(0).repeat(state.shape[0], 1, 1, 1)

        active_grids = []
        for i in range(4):
            grid = grids[:, i]
            foot_pos_i = foot_pos[:, i, :2]
            foot_pos_i = foot_pos_i.unsqueeze(1).repeat(1, self.num_bins**2, 1)
            min_xy = grid[:, :, :2]
            max_xy = min_xy + grid[:, :, 2:]

            # Check if the foot position falls within the grid squares
            active = ((foot_pos_i[:, :, 0] >= min_xy[:, :, 0]) & (foot_pos_i[:, :, 0] <= max_xy[:, :, 0]) &
                      (foot_pos_i[:, :, 1] >= min_xy[:, :, 1]) & (foot_pos_i[:, :, 1] <= max_xy[:, :, 1]))
            active = active.int()
            active_grids.append(active)

        return torch.stack(active_grids, dim=1)
            
    
    def get_avoid_grids(self, state):
        """Sets the grid index to 1 if the grid is over the gap"""
        com = state[:, None, 36:38]
        orientation = state[:, 39:48].reshape(-1, 3, 3)[:, :2, :2]
        grids = torch.stack(list(self.grid.values()))
        grids = grids.unsqueeze(0).repeat(state.shape[0], 1, 1, 1)

        active_grids = []
        for i in range(4):
            grid = grids[:, i]
            min_xy = torch.einsum('bij,bjk->bik', (grid[:, :, :2], orientation)) + com
            max_xy = torch.einsum('bij,bjk->bik', (grid[:, :, :2] + grid[:, :, 2:], orientation)) + com
            active = ((min_xy[:, :, 0] >= 3) & (min_xy[:, :, 0] <= 3.1)) | \
                ((max_xy[:, :, 0] >= 3) & (max_xy[:, :, 0] <= 3.1))
            active = active.int() 
            active_grids.append(active)
        return torch.stack(active_grids, dim=1)


def test():
    foot_grid = FootGrid(
        torch.tensor([[0.22, 0.48, 0, 0.24], 
                  [0.22, 0.48, -0.24, 0],
                  [-0.45, -0.2, 0, 0.24],
                  [-0.45, -0.2, -0.24, 0]]), 4, 'cpu')
    state = torch.tensor([[-3.19502503e-02, -3.09347957e-02,  9.99010623e-01,
        -3.69389892e-01,  5.21401405e-01, -9.46781635e-01,
         4.33597565e-02,  5.69789529e-01, -6.91575766e-01,
        -2.26425499e-01, -7.45631337e-01,  7.71101952e-01,
         4.02590305e-01, -6.14872515e-01,  7.79008329e-01,
        -1.58601142e-02,  1.47256041e-02,  1.27757758e-01,
        -1.11884743e-01,  1.30208984e-01, -1.20830648e-01,
        -7.76994973e-02,  1.05080038e-01,  7.12922066e-02,
         5.30035868e-02,  2.38051433e-02,  3.62314843e-02,
         6.68667033e-02,  9.99930501e-02,  7.44123897e-03,
         5.70536107e-02,  1.14180902e-02, -5.55264391e-03,
         2.32830644e-10,  5.82076609e-11, -1.16415322e-10,
        -2.26867461e+00,  1.39827859e+00,  5.78197122e-01,
        -6.93322599e-01, -7.19908416e-01, -3.21813971e-02,
         7.20627308e-01, -6.92642152e-01, -3.07100918e-02,
        -1.81739670e-04, -4.44827937e-02,  9.99010146e-01,
         4.36683506e-01,  9.92687941e-02, -5.54718673e-01,
         3.32761109e-01, -2.77594864e-01, -5.72587848e-01,
        -2.63536185e-01,  1.74099475e-01, -5.75333536e-01,
        -3.38565856e-01, -6.86925426e-02, -5.84918082e-01]])
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