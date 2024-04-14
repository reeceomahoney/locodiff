import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from beso.agent.utils import MinMaxScaler
from data.trajectory_loader import TrajectoryDataset, get_train_val_sliced


def get_raisim_train_val(
    data_directory,
    obs_dim,
    window_size,
    future_seq_len,
    train_fraction,
    random_seed,
    device,
    train_batch_size,
    test_batch_size,
    num_workers,
):
    train_set, test_set = get_train_val_sliced(
        RaisimTrajectoryDataset(data_directory, future_seq_len, obs_dim, window_size),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_seq_len,
    )
    scaler = MinMaxScaler(
        train_set.dataset.dataset.get_all_observations(),
        train_set.dataset.dataset.get_all_actions(),
        device,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader, scaler


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

        data = np.load(self.dataset_path, allow_pickle=True).item()
        self.observations = data["observations"]
        self.actions = data["actions"]
        self.terminals = data["terminals"]
        self.preprocess()
        self.split_data()

        self.observations = torch.from_numpy(self.observations).to(device).float()
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()

        logging.info(
            f"Dataset size - Observations: {list(self.observations.size())} - Actions: {list(self.actions.size())}"
        )
        tensors = [self.observations, self.actions, self.masks]

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
        # Add skill to the observations
        self.observations = np.concatenate(
            [self.observations, np.zeros_like(self.observations[:, :, :1])], axis=-1
        )
        B = self.observations.shape[0]
        # self.observations[: B // 2, :, -1] = 0
        # self.observations[B // 2 :, :, -1] = 1

        # Flatten the first two dimensions
        obs_flat = self.observations.reshape(-1, self.observations.shape[-1])
        actions_flat = self.actions.reshape(-1, self.actions.shape[-1])
        terminals_flat = self.terminals.reshape(-1)

        # Find the indices where terminals is True (or 1)
        split_indices = np.where(terminals_flat == 1)[0]

        # Split the flattened observations and actions into sequences
        obs_splits = np.split(obs_flat, split_indices)
        actions_splits = np.split(actions_flat, split_indices)

        # Find the maximum length of the sequences
        max_len = max(split.shape[0] for split in obs_splits)

        # Pad the sequences and reshape them back to their original shape
        self.observations = self.pad_and_stack(obs_splits, max_len).astype(np.float32)
        self.actions = self.pad_and_stack(actions_splits, max_len).astype(np.float32)
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
