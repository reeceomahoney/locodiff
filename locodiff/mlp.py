import torch
import torch.nn as nn
from .utils import SinusoidalPosEmb


class DiffusionMLPSieve(nn.Module):
    def __init__(self, obs_dim, act_dim, T, T_cond, n_emb, n_hidden):
        super(DiffusionMLPSieve, self).__init__()
        self.T = T

        # embedding
        self.cond_emb = nn.Sequential(
            nn.Linear(obs_dim + 2, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.action_emb = nn.Sequential(
            nn.Linear(act_dim, n_emb),
            nn.LeakyReLU(),
            nn.Linear(n_emb, n_emb),
        )
        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(n_emb),
            nn.Linear(n_emb, n_emb),
        )

        # decoder
        dims = [
            ((T + T_cond + 1) * n_emb, n_hidden),
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

    def forward(self, x, cond, sigma, **kwargs):
        B = x.shape[0]

        goal = kwargs["goal"].unsqueeze(1).repeat(1, cond.shape[1], 1)
        cond = torch.cat((cond, goal), dim=-1)

        cond_emb = self.cond_emb(cond)
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
