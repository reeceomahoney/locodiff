import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data_path):
    return np.load(data_path, allow_pickle=True).item()


def cat_zeros(x):
    return np.concatenate((x, np.zeros_like(x[:, :, 0:1])), axis=-1)


def cat(x1, x2):
    return {k: np.concatenate((x1[k], x2[k]), axis=0) for k in x1.keys()}


def shift_terminals(terminals):
    """Shifts the terminal states to the next time step. This is the format expected by the dataloader"""
    term_x, term_y, _ = np.where(terminals == 1)
    term_y = term_y + 1

    # Remove overflows
    valid_indices = term_y < terminals.shape[1]
    term_x = term_x[valid_indices]
    term_y = term_y[valid_indices]

    terminals[...] = 0
    terminals[term_x, term_y] = 1

    # add a terminal for every episode start except the first one
    terminals[:, 0] = 1
    terminals[0, 0] = 0
    return terminals


def split_by_vel_cmd(obs, terminals):
    """Adds terminals at timesteps where the velocity command changes"""
    diff = np.diff(obs[..., -3:], axis=1)
    diff_idx_x, diff_idx_y = np.where(np.any(diff != 0, axis=-1))
    diff_idx_y += 1
    terminals[diff_idx_x, diff_idx_y, 0] = 1
    return terminals


current_dir = os.path.dirname(os.path.realpath(__file__))
walk_data = load_data(current_dir + "/datasets/raw/walk_rand_slow.npy")
# crawl_data = load_data(current_dir + "/datasets/raw/crawl.npy")
# data = cat(walk_data, crawl_data)
data = walk_data

obs, act, terminals = data["observations"], data["actions"], data["terminals"]
terminals = shift_terminals(terminals)

# obs = cat_zeros(obs)
# obs[:1000, :, -1] = 1
# obs[1000:, :, -1] = -1

# Save the data to a new file
name = "walk_rand_slow"
print(f"Saving data to {current_dir}/datasets/{name}.npy")
print(f"Observations shape: {obs.shape}, Actions shape: {act.shape}")
np.save(
    f"{current_dir}/datasets/{name}.npy",
    {"observations": obs, "actions": act, "terminals": terminals},
)
print("done")
