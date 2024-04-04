import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data_path):
    data = np.load(data_path, allow_pickle=True).item()
    return data["observations"], data["actions"], data["terminals"]


current_dir = os.path.dirname(os.path.realpath(__file__))
walk_obs, walk_act, walk_terminals = load_data(
    current_dir + "/datasets/vel_cmd/fwd_25.npy"
)
crawl_obs, crawl_act, crawl_terminals = load_data(
    current_dir + "/datasets/pos_cmd_3/crawl.npy"
)

cat = lambda x, y: np.concatenate((x, y), axis=0)
# obs = cat(walk_obs, crawl_obs)
# act = cat(walk_act, crawl_act)
# terminals = cat(walk_terminals, crawl_terminals)
obs = walk_obs
act = walk_act
terminals = walk_terminals

# move terminals to after the episode end
term_x, term_y, _ = np.where(terminals == 1)
term_y = np.minimum(term_y + 1, 999)
terminals[..., 0] = 0
terminals[term_x, term_y, 0] = 1
terminals[:, -1, 0] = 0

# split episodes by velocity commands
# diff = np.diff(obs[..., -3:], axis=1)
# diff_idx_x, diff_idx_y = np.where(np.any(diff != 0, axis=-1))
# diff_idx_y += 1
# terminals[diff_idx_x, diff_idx_y, 0] = 1

# obs = obs[..., :35]
print(obs.shape, act.shape)

# Save the concatenated data to a new file
np.save(
    current_dir + "/datasets/fwd.npy",
    {"observations": obs, "actions": act, "terminals": terminals},
)
print("done")
