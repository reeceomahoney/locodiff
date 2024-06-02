import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Union, Optional, Sequence, List
import abc
from torch import default_generator, randperm
from itertools import accumulate


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.
    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        observations, actions, masks = self.dataset[self.indices]

        result = []
        # mask out invalid actions
        for i in range(len(masks)):
            T = int(masks[i].sum().item())
            result.append(actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        observations, actions, masks = self.dataset[self.indices]

        result = []
        # mask out invalid actions
        for i in range(len(masks)):
            T = int(masks[i].sum().item())
            result.append(observations[i, :T, :])
        return torch.cat(result, dim=0)


class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_seq_len: int,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.
        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                0: invalid
                1: valid
        window: int
            number of timesteps to include in each slice
        future_seq_len: int
            the length of the future conditional sequence;
        """
        self.dataset = dataset
        self.window = window
        self.future_seq_len = future_seq_len
        self.slices = []
        min_seq_length = np.inf
        effective_window = window + future_seq_len - 1

        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - effective_window < 0:
                # print(f"Ignored short sequence #{i}: len={T}, window={effective_window}")
                pass
            else:
                self.slices += [
                    (i, start, start + effective_window)
                    for start in range(T - effective_window + 1)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < effective_window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        return self.future_seq_len + self.window

    def get_completed_goals(self) -> torch.Tensor:
        goals = []

    def get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        data_batch = {}
        i, start, end = self.slices[idx]
        values = [x[start:end] for x in self.dataset[i]]
        data_batch["observation"] = values[0]
        data_batch["action"] = values[1]
        data_batch["cmd"] = self.dataset[i][2]
        data_batch["indicator"] = self.dataset[i][3]

        return data_batch


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.95,
    random_seed: int = 42,
    device: Union[str, torch.device] = "cpu",
    window_size: int = 10,
    future_seq_len: int = 10,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_seq_len": future_seq_len,
    }
    if window_size > 0:
        train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
        val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
        return train_slices, val_slices
    else:
        return train, val


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)
    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:
    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set
