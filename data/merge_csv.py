import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Get a list of all CSV files in the directory
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = current_dir + "/datasets/2024-03-01-14-07-05"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Load each CSV file into a separate numpy array
arrays = [pd.read_csv(os.path.join(data_dir, f), header=None).values for f in csv_files]

# Concatenate all arrays
data = np.stack(arrays, axis=0)[..., 1:]
data = data.reshape(-1, 1000, 48)

crawl_obs = data[:, :, :36]
crawl_act = data[:, :, 36:]

walk_data = np.load(
    current_dir + "/datasets/rand_25.npy", allow_pickle=True
).item()

walk_obs = walk_data["observations"][:, :, :36]
walk_act = walk_data["actions"]

obs = np.concatenate((walk_obs, crawl_obs), axis=0)
act = np.concatenate((walk_act, crawl_act), axis=0)
terminals = np.zeros((obs.shape[0], obs.shape[1], 1))
# obs = walk_obs
# act = walk_act

print(obs.shape, act.shape)

# Save the concatenated data to a new file
np.save(
    current_dir + "/datasets/walk_crawl.npy",
    {"observations": obs, "actions": act, "terminals": terminals},
)
print("done")
