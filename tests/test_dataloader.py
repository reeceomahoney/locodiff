import unittest
import numpy as np
import torch
from beso.envs.raisim.data.dataloader import RaisimTrajectoryDataset


class TestDataloader(unittest.TestCase):
    def test_split_data(self):
        # Create dummy data
        np.random.seed(42)
        data_directory = "/home/reece/Workspace/Projects/beso/beso/envs/raisim/data/random_acts.npy"
        obs = np.random.rand(2, 10, 3)
        actions = np.random.rand(2, 10, 2)
        terminals = np.zeros((2, 10))
        terminals[0, 5] = 1
        terminals[1, 7] = 1
        terminals[:, 0] = 1
        terminals[0, 0] = 0

        dataset = RaisimTrajectoryDataset(data_directory)
        dataset.observations = obs
        dataset.actions = actions
        dataset.terminals = terminals
        dataset.split_data()


        # Check the results
        expected_obs = np.zeros((4, 7, 3))
        expected_obs[0, :5, :] = obs[0, :5, :]
        expected_obs[1, :5, :] = obs[0, 5:10, :]
        expected_obs[2, :7, :] = obs[1, :7, :]
        expected_obs[3, :3, :] = obs[1, 7:10, :]

        expected_actions = np.zeros((4, 7, 2))
        expected_actions[0, :5, :] = actions[0, :5, :]
        expected_actions[1, :5, :] = actions[0, 5:10, :]
        expected_actions[2, :7, :] = actions[1, :7, :]
        expected_actions[3, :3, :] = actions[1, 7:10, :]

        expected_masks = np.zeros((4, 7))
        expected_masks[0, :5] = 1
        expected_masks[1, :5] = 1
        expected_masks[2, :7] = 1
        expected_masks[3, :3] = 1

        self.assertTrue(np.allclose(dataset.observations, expected_obs))
        self.assertTrue(np.allclose(dataset.actions, expected_actions))
        self.assertTrue(np.allclose(dataset.masks, expected_masks))

if __name__ == "__main__":
    unittest.main()
