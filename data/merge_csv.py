import numpy as np
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))

crawl_data = np.load(
    current_dir + "/datasets/lfmc_crawl.npy", allow_pickle=True
).item()

crawl_obs = crawl_data["observations"]
crawl_act = crawl_data["actions"]
crawl_terminals = crawl_data["terminals"]

walk_data = np.load(
    current_dir + "/datasets/rand_25.npy", allow_pickle=True
).item()

walk_obs = walk_data["observations"][:, :, :36]
walk_act = walk_data["actions"]
walk_terminals = walk_data["terminals"]

obs = np.concatenate((walk_obs, crawl_obs), axis=0)
act = np.concatenate((walk_act, crawl_act), axis=0)
terminals = np.concatenate((walk_terminals, crawl_terminals), axis=0)

# split episodes by velocity commands
diff = np.diff(obs[..., -3:], axis=1)
diff_idx_x, diff_idx_y = np.where(np.any(diff != 0, axis=-1))
diff_idx_y += 1
terminals[diff_idx_x, diff_idx_y, 0] = 1

print(obs.shape, act.shape)

# Save the concatenated data to a new file
np.save(
    current_dir + "/datasets/lfmc_walk_crawl.npy",
    {"observations": obs, "actions": act, "terminals": terminals},
)
print("done")
