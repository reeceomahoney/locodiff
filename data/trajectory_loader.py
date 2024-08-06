import torch
from torch.utils.data import Dataset


class TrajectorySlicerDataset(Dataset):
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


def split_dataset(
    dataset: Dataset, train_fraction: float = 0.95, random_seed: int = 42
):
    dataset_size = len(dataset)
    train_size = int(train_fraction * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset


def get_train_val_sliced(
    dataset: Dataset,
    train_fraction: float = 0.95,
    random_seed: int = 42,
    window_size: int = 10,
    future_seq_len: int = 10,
):
    train, val = split_dataset(dataset, train_fraction, random_seed)
    if window_size > 0:
        return (
            TrajectorySlicerDataset(train, window_size, future_seq_len),
            TrajectorySlicerDataset(val, window_size, future_seq_len),
        )
    return train, val
