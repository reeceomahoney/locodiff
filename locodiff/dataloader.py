import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from locodiff.utils import MinMaxScaler


class ExpertDataset(Dataset):
    def __init__(
        self,
        data_directory: str,
        obs_dim: int,
        T_cond: int,
        return_horizon: int,
        device="cpu",
    ):
        self.data_directory = data_directory
        self.obs_dim = obs_dim
        self.T_cond = T_cond
        self.return_horizon = return_horizon
        self.device = device

        current_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_path = current_dir + "/../data/" + data_directory + ".npy"

        self.data = self.load_and_process_data(dataset_path)

        obs_size = list(self.data["obs"].shape)
        action_size = list(self.data["action"].shape)
        print(f"Dataset size | Observations: {obs_size} | Actions: {action_size}")

    # --------------
    # Initialization
    # --------------

    def load_and_process_data(self, dataset_path):
        data = np.load(dataset_path, allow_pickle=True).item()

        obs = data["obs"][..., :33]
        actions = data["action"]
        vel_cmds = data["vel_cmd"]
        skills = data["skill"]
        terminals = data["terminal"]
        # torques = data["torque"]

        # Find episode ends
        terminals_flat = terminals.reshape(-1)
        split_indices = np.where(terminals_flat == 1)[0]

        # Split the sequences at episode ends
        obs_splits = self.split_eps(obs, split_indices)
        actions_splits = self.split_eps(actions, split_indices)
        vel_cmds_splits = self.split_eps(vel_cmds, split_indices)
        skills_splits = self.split_eps(skills, split_indices)
        # torques_splits = self.split_eps(torques, split_indices)

        max_len = max(split.shape[0] for split in obs_splits)

        obs = self.add_padding(obs_splits, max_len, temporal=True)
        actions = self.add_padding(actions_splits, max_len, temporal=True)
        vel_cmds = self.add_padding(vel_cmds_splits, max_len, temporal=False)
        skills = self.add_padding(skills_splits, max_len, temporal=True)
        # torques = self.add_padding(torques_splits, max_len, temporal=True)

        # NB: skill initial pad is the same as unconditional mask. this might be a problem.

        masks = self.create_masks(obs_splits, max_len)

        # Compute returns
        if self.return_horizon > 0:
            vel_cmds = self.sample_vel_cmd(obs.shape[0])
            returns, rewards = self.compute_returns(obs, vel_cmds, masks)

            # Remove last steps if return horizon is set
            obs = obs[:, : -self.return_horizon]
            actions = actions[:, : -self.return_horizon]
            skills = skills[:, : -self.return_horizon]
            terminals = terminals[:, : -self.return_horizon]
            masks = masks[:, : -self.return_horizon]

        processed_data = {
            "obs": obs,
            "action": actions,
            "vel_cmd": vel_cmds,
            "skill": skills,
            "return": returns,
            # "reward": rewards,
            "mask": masks,
        }

        return processed_data

    # -------
    # Getters
    # -------

    def __len__(self):
        return len(self.data["obs"])

    def __getitem__(self, idx):
        T = self.data["mask"][idx].sum().int().item()
        return {
            key: tensor[idx, :T] for key, tensor in self.data.items() if key != "mask"
        }

    def get_seq_length(self, idx):
        return int(self.data["mask"][idx].sum().item())

    def get_all_obs(self):
        return torch.cat(
            [self.data["obs"][i, : self.get_seq_length(i)] for i in range(len(self))]
        )

    def get_all_actions(self):
        return torch.cat(
            [self.data["action"][i, : self.get_seq_length(i)] for i in range(len(self))]
        )

    def get_all_vel_cmds(self):
        return self.data["vel_cmd"].flatten()

    # ----------------
    # Helper functions
    # ----------------

    def split_eps(self, x, split_indices):
        return np.split(x.reshape(-1, x.shape[-1]), split_indices)

    def add_padding(self, splits, max_len, temporal):
        x = []

        # Make all sequences the same length
        for split in splits:
            pad = np.pad(split, ((0, max_len - split.shape[0]), (0, 0)))
            reshaped_split = pad.reshape(-1, max_len, split.shape[1])
            x.append(reshaped_split)
        x = np.stack(x).squeeze(axis=1)

        if temporal:
            # Add initial padding to handle episode starts
            x_pad = np.zeros_like(x[:, : self.T_cond - 1, :])
            x = np.concatenate([x_pad, x], axis=1)
        else:
            # For non-temporal data, e.g. skills, just take the first timestep
            x = x[:, 0]

        return torch.from_numpy(x).to(self.device).float()

    def create_masks(self, splits, max_len):
        masks = []

        # Create masks to indicate the padding values
        for split in splits:
            mask = np.concatenate(
                [np.ones(split.shape[0]), np.zeros(max_len - split.shape[0])]
            )
            masks.append(mask)
        masks = np.stack(masks)

        # Add initial padding to handle episode starts
        masks_pad = np.ones((masks.shape[0], self.T_cond - 1))
        masks = np.concatenate([masks_pad, masks], axis=1)

        return torch.from_numpy(masks).to(self.device).float()

    def sample_vel_cmd(self, batch_size):
        vel_cmd = torch.randint(0, 2, (batch_size, 1), device=self.device)
        vel_cmd = vel_cmd.float() * 2 - 1

        return vel_cmd

    def compute_returns(self, obs, vel_cmds, masks):
        lin_vel = obs[..., 30:32]
        ang_vel = obs[..., 17:18]
        vel = torch.cat([lin_vel, ang_vel], dim=-1)

        vel = vel[:, :, 0]
        vel_cmds = vel_cmds.expand(-1, vel.shape[1])

        rewards = torch.zeros_like(vel)
        rewards = torch.where(vel_cmds == 1, vel, rewards)
        rewards = torch.where(vel_cmds == -1, -vel, rewards)
        rewards = torch.clamp(rewards, -0.6, 0.6)
        rewards -= rewards.max()

        horizon = self.return_horizon
        gammas = torch.tensor([0.99**i for i in range(horizon)]).to(self.device)
        returns = torch.zeros_like(rewards[:, :-horizon])

        for i in range(masks.shape[0]):
            T = int(masks[i].sum().item())
            for t in range(T - horizon):
                returns[i, t] = (rewards[i, t : t + horizon] * gammas).sum()

        returns = torch.exp(returns/20)
        return returns.unsqueeze(-1), rewards


class SlicerWrapper(Dataset):
    def __init__(self, dataset: Subset, T_cond: int, T: int):
        self.dataset = dataset
        self.T_cond = T_cond
        self.T = T
        self.slices = self._create_slices(T_cond, T)

    def _create_slices(self, T_cond, T):
        slices = []
        window = T_cond + T - 1
        for i in range(len(self.dataset)):
            length = len(self.dataset[i]["obs"])
            if length >= window:
                slices += [
                    (i, start, start + window) for start in range(length - window + 1)
                ]
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        x = self.dataset[i]

        # This is to handle data without a time dimension (e.g. skills)
        return {k: v[start:end] if v.ndim > 1 else v for k, v in x.items()}

    def get_all_obs(self):
        return self.dataset.dataset.get_all_obs()

    def get_all_actions(self):
        return self.dataset.dataset.get_all_actions()

    def get_all_vel_cmds(self):
        return self.dataset.dataset.get_all_vel_cmds()


def get_dataloaders_and_scaler(
    data_directory: str,
    obs_dim: int,
    T_cond: int,
    T: int,
    train_fraction: float,
    device: str,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    return_horizon: int,
):
    # Build the datasets
    dataset = ExpertDataset(data_directory, obs_dim, T_cond, return_horizon)
    train, val = random_split(dataset, [train_fraction, 1 - train_fraction])
    train_set = SlicerWrapper(train, T_cond, T)
    test_set = SlicerWrapper(val, T_cond, T)

    # Build the scaler
    x_data = train_set.get_all_obs()
    y_data = train_set.get_all_actions()
    scaler = MinMaxScaler(x_data, y_data, device)

    # Build the dataloaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, scaler
