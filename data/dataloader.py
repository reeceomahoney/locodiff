import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from locodiff.utils import MinMaxScaler

log = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        future_seq_len: int,
        obs_dim: int,
        T_cond: int,
        device="cpu",
    ):
        self.data_directory = data_directory
        self.obs_dim = obs_dim
        self.T_cond = T_cond
        self.future_seq_len = future_seq_len
        self.device = device

        current_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_path = current_dir + "/datasets/" + data_directory + ".npy"
        self.dataset_path = Path(dataset_path)

        self.load_and_process_data()

        log.info(
            f"Dataset size - Observations: {self.data['obs'].shape} - Actions: {self.data['action'].shape}"
        )

    def load_and_process_data(self):
        data = np.load(self.dataset_path, allow_pickle=True).item()

        obs, actions, vel_cmds, masks = self.process_raw_data(data)

        # (batch, time, features)
        self.data = {
            "obs": obs,
            "action": actions,
            "vel_cmd": vel_cmds,
            "mask": masks,
            "indicator": vel_cmds[:, :2],
        }

    def process_raw_data(self, data):
        obs = data["observations"]
        actions = data["actions"]
        vel_cmds = data["vel_cmds"]
        terminals = data["terminals"]

        # Flatten the first two dimensions
        obs_flat = obs.reshape(-1, obs.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        cmd_flat = vel_cmds.reshape(-1, vel_cmds.shape[-1])
        terminals_flat = terminals.reshape(-1)

        # Find the indices where terminals is True (or 1)
        split_indices = np.where(terminals_flat == 1)[0]

        # Split the flattened observations and actions into sequences
        obs_splits = np.split(obs_flat, split_indices)
        actions_splits = np.split(actions_flat, split_indices)
        cmd_splits = np.split(cmd_flat, split_indices)

        # Find the maximum length of the sequences
        max_len = max(split.shape[0] for split in obs_splits)

        # Pad the sequences and reshape them back to their original shape
        obs = self.pad_and_stack(obs_splits, max_len).astype(np.float32)
        actions = self.pad_and_stack(actions_splits, max_len).astype(np.float32)
        vel_cmds = self.pad_and_stack(cmd_splits, max_len).astype(np.float32)
        masks = self.create_masks(obs_splits, max_len)

        # Add initial padding to handle episode starts
        obs_initial_pad = np.zeros_like(obs[:, : self.T_cond - 1, :])
        obs = np.concatenate([obs_initial_pad, obs], axis=1)

        actions_initial_pad = np.zeros_like(actions[:, : self.T_cond - 1, :])
        actions = np.concatenate([actions_initial_pad, actions], axis=1)

        masks_initial_pad = np.ones((masks.shape[0], self.T_cond - 1))
        masks = np.concatenate([masks_initial_pad, masks], axis=1)

        vel_cmds = vel_cmds[:, 0, :3]

        obs = torch.from_numpy(obs).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).float()
        masks = torch.from_numpy(masks).to(self.device).float()
        vel_cmds = torch.from_numpy(vel_cmds).to(self.device).float()

        return obs, actions, vel_cmds, masks

    def __len__(self):
        return len(self.data["obs"])

    def __getitem__(self, idx):
        T = self.data["mask"][idx].sum().int().item()
        return {
            key: tensor[idx, :T] for key, tensor in self.data.items() if key != "mask"
        }

    def get_seq_length(self, idx):
        return int(self.data["mask"][idx].sum().item())

    def get_all_actions(self):
        return torch.cat(
            [self.data["action"][i, : self.get_seq_length(i)] for i in range(len(self))]
        )

    def get_all_observations(self):
        return torch.cat(
            [self.data["obs"][i, : self.get_seq_length(i)] for i in range(len(self))]
        )

    def get_all_vel_cmds(self):
        return self.data["vel_cmd"].flatten(0, 1)

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


class SlicerWrapper(Dataset):
    def __init__(self, dataset, window: int, future_seq_len: int):
        self.dataset = dataset
        self.window = window
        self.future_seq_len = future_seq_len
        self.slices = self._create_slices()

    def _create_slices(self):
        slices = []
        effective_window = self.window + self.future_seq_len - 1 + 49
        for i in range(len(self.dataset)):
            T = len(self.dataset[i]["obs"])
            if T >= effective_window:
                slices += [
                    (i, start, start + effective_window)
                    for start in range(T - effective_window + 1)
                ]
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        x = self.dataset[i]
        return {k: v[start:end] if v.ndim > 1 else v for k, v in x.items()}


def get_dataloaders_and_scaler(
    data_directory,
    obs_dim,
    T_cond,
    T,
    train_fraction,
    device,
    train_batch_size,
    test_batch_size,
    num_workers,
):
    # Build the datasets
    dataset = ExpertDataset(data_directory, T, obs_dim, T_cond)
    train, val = torch.utils.data.random_split(
        dataset, [train_fraction, 1 - train_fraction]
    )
    train_set = SlicerWrapper(train, T_cond, T)
    test_set = SlicerWrapper(val, T_cond, T)

    # Build the scaler
    x_data = train_set.dataset.dataset.get_all_observations()
    y_data = train_set.dataset.dataset.get_all_actions()
    cmd_data = train_set.dataset.dataset.get_all_vel_cmds()
    scaler = MinMaxScaler(x_data, y_data, cmd_data, device)

    # Build the dataloaders
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
