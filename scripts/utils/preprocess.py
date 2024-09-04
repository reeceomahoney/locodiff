import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data_path):
    return np.load(data_path, allow_pickle=True).item()


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


data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../data/"
walk_data = load_data(data_dir + "raw/noise/walk_pos_noise.npy")
# crawl_data = load_data(data_dir + "raw/crawl.npy")

# roll actions (only need this for pmtg)
act = np.roll(walk_data["actions"], -1, axis=1)
act[:, -1] = act[:, -2].copy()
walk_data["actions"] = act.copy()

# data = cat(walk_data, crawl_data)
data = walk_data

obs = data["observations"]
act = data["actions"]
terminals = data["terminals"]
vel_cmds = data["vel_cmds"]

terminals = shift_terminals(terminals)

skill = np.zeros_like(vel_cmds[..., :2])
skill[:1000, :, 0] = 1
# skill[1000:, :, 1] = 1

processed_data = {
    "obs": obs,
    "action": act,
    "vel_cmd": vel_cmds,
    "skill": skill,
    "terminal": terminals,
}

# Save the data to a new file
name = "noise/walk_pos_noise"
print(f"Saving data to {data_dir}/{name}.npy")
print(f"Observations shape: {obs.shape}, Actions shape: {act.shape}")
np.save(f"{data_dir}/{name}.npy", processed_data)
print("done")
