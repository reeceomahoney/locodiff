import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data_path):
    return np.load(data_path, allow_pickle=True).item()


def cat_zeros(x):
    return np.concatenate((x, np.zeros_like(x[:, :, 0:2])), axis=-1)


def cat(x, name):
    return np.concatenate([x[key][name] for key in data.keys()], axis=0)


def shift_terminals(terminals):
    """Shifts the terminal states to the next time step. This is the format expected by the dataloader"""
    term_x, term_y, _ = np.where(terminals == 1)
    term_y = np.minimum(term_y + 1, terminals.shape[1] - 1)
    terminals[..., 0] = 0
    terminals[term_x, term_y, 0] = 1
    terminals[:, -1, 0] = 0

    # add a terminal for every episode start
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
data = load_data(current_dir + "/datasets/raw/heading_2.npy")

obs, act, terminals = data["observations"], data["actions"], data["terminals"]
terminals = shift_terminals(terminals)

# Save the data to a new file
name = "heading"
print(f"Saving data to {current_dir}/datasets/{name}.npy")
print(f"Observations shape: {obs.shape}, Actions shape: {act.shape}")
np.save(
    f"{current_dir}/datasets/{name}.npy",
    {"observations": obs, "actions": act, "terminals": terminals},
)
print("done")
