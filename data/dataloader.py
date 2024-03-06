from typing import Optional
import logging

import os
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
import numpy as np

from data.trajectory_loader import (
    TrajectoryDataset,
    get_train_val_sliced,
)


def get_raisim_train_val(
    data_directory,
    obs_dim,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    only_sample_tail: bool = False,
    only_sample_seq_end: bool = False,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]

    return get_train_val_sliced(
        RaisimTrajectoryDataset(data_directory, future_seq_len, obs_dim, window_size),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        only_sample_tail=only_sample_tail,
        only_sample_seq_end=only_sample_seq_end,
    )


class RaisimTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        future_seq_len: int,
        obs_dim: int,
        T_cond: int,
        device="cpu",
    ):
        self.device = device
        dataset_path = (
            os.path.dirname(os.path.realpath(__file__))
            + "/datasets/"
            + data_directory
            + ".npy"
        )
        self.dataset_path = Path(dataset_path)
        self.data_directory = data_directory
        self.obs_dim = obs_dim
        self.T_cond = T_cond
        self.future_seq_len = future_seq_len
        logging.info("Data loading: started")
        data = np.load(self.dataset_path, allow_pickle=True).item()
        self.observations = data["observations"]
        self.actions = data["actions"]
        self.terminals = data["terminals"]
        self.preprocess()
        self.split_data()

        self.observations = torch.from_numpy(self.observations).to(device).float()
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.goals = torch.from_numpy(self.goals).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()

        logging.info("Data loading: done")
        tensors = [self.observations, self.actions, self.masks, self.goals]

        # The current values are in shape N x T x Dim, so all is good in the world.
        TensorDataset.__init__(self, *tensors)

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def preprocess(self):
        if self.data_directory == "rand_feet" or self.data_directory == "rand_feet_com":
            self.observations = np.concatenate(
                [self.observations[:, :, :36], self.observations[:, :, 48:]], axis=-1
            )
        else:
            self.observations = self.observations[..., : self.obs_dim]

        # To split episodes correctly
        self.terminals[:, 0] = 1
        self.terminals[0, 0] = 0

    def split_data(self):
        # generate one-hot goals
        self.goals = np.zeros_like(self.terminals)
        self.goals[: self.terminals.shape[0] // 2] = 1
        self.goals[self.terminals.shape[0] // 2 :] = -1

        # Flatten the first two dimensions
        obs_flat = self.observations.reshape(-1, self.observations.shape[-1])
        actions_flat = self.actions.reshape(-1, self.actions.shape[-1])
        terminals_flat = self.terminals.reshape(-1)
        goals_flat = self.goals.reshape(-1, 1)

        # Find the indices where terminals is True (or 1)
        split_indices = np.where(terminals_flat == 1)[0]

        # Split the flattened observations and actions into sequences
        obs_splits = np.split(obs_flat, split_indices)
        actions_splits = np.split(actions_flat, split_indices)
        goal_splits = np.split(goals_flat, split_indices)

        # Find the maximum length of the sequences
        max_len = max(split.shape[0] for split in obs_splits)

        # Pad the sequences and reshape them back to their original shape
        self.observations = self.pad_and_stack(obs_splits, max_len).astype(np.float32)
        self.actions = self.pad_and_stack(actions_splits, max_len).astype(np.float32)
        self.goals = self.pad_and_stack(goal_splits, max_len)
        self.masks = self.create_masks(obs_splits, max_len)

        # Add initial padding to handle episode starts
        obs_initial_pad = np.repeat(
            self.observations[:, 0:1, :], self.T_cond - 1, axis=1
        )
        self.observations = np.concatenate([obs_initial_pad, self.observations], axis=1)

        actions_initial_pad = np.zeros(
            (self.actions.shape[0], self.T_cond - 1, self.actions.shape[2])
        )
        self.actions = np.concatenate([actions_initial_pad, self.actions], axis=1)

        goals_initial_pad = np.concatenate(
            [
                np.ones((self.goals.shape[0] // 2, self.T_cond - 1)),
                -1 * np.ones((self.goals.shape[0] // 2, self.T_cond - 1)),
            ]
        )
        # goals_initial_pad = np.zeros((self.goals.shape[0], self.T_cond-1))
        self.goals = np.concatenate([goals_initial_pad, self.goals], axis=1)

        masks_initial_pad = np.ones((self.masks.shape[0], self.T_cond - 1))
        self.masks = np.concatenate([masks_initial_pad, self.masks], axis=1)

    def pad_and_stack(self, splits, max_len):
        """Pad the sequences and stack them into a tensor"""
        return np.stack(
            [
                np.pad(
                    split, ((0, max_len - split.shape[0]), (0, 0)), mode="constant"
                ).reshape(-1, max_len, split.shape[1])
                for split in splits
            ]
        ).squeeze()

    def create_masks(self, splits, max_len):
        """Create masks for the sequences"""
        return np.stack(
            [
                np.pad(
                    np.ones(split.shape[0]),
                    (0, max_len - split.shape[0]),
                    mode="constant",
                )
                for split in splits
            ]
        )

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)
