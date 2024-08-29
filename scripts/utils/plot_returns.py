import matplotlib.pyplot as plt
from locodiff.dataloader import ExpertDataset

data_directory = "walk"
obs_dim = 33
T_cond = 8
return_horizon = 50
reward_fn = "vel_target"

dataset = ExpertDataset(data_directory, obs_dim, T_cond, return_horizon, reward_fn)
rewards = dataset.rewards
returns = dataset.data["return"]

# Plot histogram for rewards
plt.subplot(1, 2, 1)
plt.hist(rewards.flatten().cpu().numpy(), bins=20)
plt.xlabel("Rewards")
plt.ylabel("Frequency")

# Plot histogram for returns
plt.subplot(1, 2, 2)
returns = returns[returns != 1]
print(returns.max())
plt.hist(returns.flatten().cpu().numpy(), bins=20)
plt.xlabel("Returns")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("histogram.png")

