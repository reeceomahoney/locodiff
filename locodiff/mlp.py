import torch
import torch.nn as nn

from .utils import SinusoidalPosEmb


class DiffusionMLPSieve(nn.Module):
    def __init__(self, obs_dim, act_dim, T, T_cond, n_emb, n_hidden):
        super(DiffusionMLPSieve, self).__init__()
        self.T = T
        self.cond_mask_prob = 0.1

        # embedding
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim + 1, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.action_emb = nn.Sequential(
            nn.Linear(act_dim, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.return_emb = nn.Sequential(
            nn.Linear(1, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(n_emb),
            nn.Linear(n_emb, n_emb),
        )

        # decoder
        dims = [
            ((T + T_cond + 2) * n_emb, n_hidden),
            (n_hidden + T * (act_dim) + 1, n_hidden),
            (n_hidden + T * (act_dim) + 1, n_hidden),
            (n_hidden + T * (act_dim) + 1, T * (act_dim)),
        ]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(nn.Linear(*dims[i]), nn.GELU()))
        layers.append(nn.Linear(*dims[-1]))
        self.model = nn.ModuleList(layers)

    def get_optim_groups(self, weight_decay):
        return [{"params": self.parameters()}]

    def forward(self, x, sigma, data_dict, uncond=False):
        B = x.shape[0]

        obs = data_dict["obs"]
        vel_cmd = data_dict["vel_cmd"].unsqueeze(1).expand(-1, obs.shape[1], -1)
        obs = torch.cat([obs, vel_cmd], dim=-1)
        obs_emb = self.obs_emb(obs)

        returns = self.mask_cond(data_dict["return"], uncond)
        return_emb = self.return_emb(returns).unsqueeze(1)

        cond_emb = torch.cat([return_emb, obs_emb], dim=1)

        x_emb = self.action_emb(x)
        sigma_emb = self.sigma_emb(sigma)

        # decoder
        cond_emb = cond_emb.reshape(B, -1)
        x_emb = x_emb.reshape(B, -1)
        x = x.reshape(B, -1)
        sigma = sigma.reshape(B, 1)
        out = torch.cat([cond_emb, x_emb, sigma_emb], dim=-1)
        out = self.model[0](out)
        out = self.model[1](torch.cat([out / 1.414, x, sigma], dim=-1)) + out / 1.414
        out = self.model[2](torch.cat([out / 1.414, x, sigma], dim=-1)) + out / 1.414
        out = self.model[3](torch.cat([out, x, sigma], dim=-1))
        return out.reshape(B, self.T, -1)

    def mask_cond(self, cond, force_mask=False):
        cond = cond.clone()
        if force_mask:
            cond[...] = 0
            return cond
        elif self.training and self.cond_mask_prob > 0:
            mask = (torch.rand(cond.shape[0], 1) > self.cond_mask_prob).float()
            mask = mask.expand_as(cond)
            cond[mask == 0] = 0
            return cond
        else:
            return cond


def test():
    n_emb = 128
    model = DiffusionMLPSieve(36, 12, 4, 8, 128, 4 * 128)
    x = torch.randn(4, 4, 12)
    cond = torch.randn(4, 8, 36)
    sigma = torch.rand(4)
    goal = torch.randn(4, 2)
    out = model(x, cond, sigma, goal=goal)
    print(out.shape)


if __name__ == "__main__":
    test()
