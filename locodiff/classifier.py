import torch
from torch import nn
from .utils import SinusoidalPosEmb


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, value_mean, value_std, device):
        super(ClassifierMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Mish(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Mish(),
            nn.Linear(hidden_dims[1], 1),
        )
        self.value_mean = value_mean
        self.value_std = value_std
        self.to(device)

    def forward(self, x, cond, goal):
        cond = cond.view(cond.size(0), -1).unsqueeze(1)
        cond = cond.repeat(1, x.size(1), 1)
        goal = goal.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, cond, goal), dim=-1)
        return self.model(x).squeeze(-1)

    def normalize(self, value):
        return (value - self.value_mean) / self.value_std

    def denormalize(self, value):
        return value * self.value_std + self.value_mean

    def predict(self, x, cond, goal):
        v = self(x, cond, goal)
        return self.denormalize(v)


class ClassifierTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        T,
        T_cond,
        value_mean,
        value_std,
        device,
    ):
        super(ClassifierTransformer, self).__init__()

        self.x_emb = nn.Linear(input_dim, d_model)
        self.cond_emb = nn.Linear(input_dim - 12, d_model)
        self.goal_emb = nn.Linear(2, d_model)

        self.pos_emb = (
            SinusoidalPosEmb(d_model)(torch.arange(T + 1)).unsqueeze(0).to(device)
        )
        self.cond_pos_emb = (
            SinusoidalPosEmb(d_model)(torch.arange(T_cond + 1)).unsqueeze(0).to(device)
        )

        self.encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model)
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        mask = self.generate_mask(T + 1)
        self.register_buffer("mask", mask)

        self.ln_f = nn.LayerNorm(d_model)
        self.pred = nn.Linear(d_model, 1)

        self.value_mean = value_mean
        self.value_std = value_std

        self.to(device)

    def forward(self, x, cond, goal):
        cond = self.cond_emb(cond)
        goal = self.goal_emb(goal).unsqueeze(1)
        cond = torch.cat((cond, goal), dim=1)
        cond = cond + self.cond_pos_emb

        x = self.x_emb(x)
        x = x + self.pos_emb

        x = self.encoder(x)
        x = self.decoder(tgt=x, memory=cond, tgt_mask=self.mask)
        x = self.ln_f(x)
        x = self.pred(x)

        return x.squeeze(-1)

    def normalize(self, value):
        return (value - self.value_mean) / self.value_std

    def denormalize(self, value):
        return value * self.value_std + self.value_mean

    def predict(self, x, cond, goal):
        v = self(x, cond, goal)
        return self.denormalize(v)

    def generate_mask(self, x):
        mask = (torch.triu(torch.ones(x, x)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class ClassifierGuidedSampleModel(nn.Module):
    """
    A wrapper model that adds guided conditional sampling capabilities to an existing model.

    Args:
        model (nn.Module): The underlying model to run.
        cond_func (callable): A function that provides conditional guidance.
        cond_lambda (float): Optional. The conditional lambda value. Defaults to 2.

    Attributes:
        model (nn.Module): The underlying model.
        guide (callable): The conditional guidance function.
        cond_lambda (float): The conditional lambda value.

    """

    def __init__(self, model, cond_func, cond_lambda: float = 2):
        super().__init__()
        self.model = model
        self.guide = cond_func
        self.cond_lambda = cond_lambda

    def forward(self, x_t, cond, sigma, **kwargs):
        cond_lambda = kwargs.get("cond_lambda", self.cond_lambda)
        out = self.model(x_t, cond, sigma, **kwargs)
        with torch.enable_grad():
            x = out.clone().requires_grad_(True)
            q_value = self.guide.predict(x, cond, kwargs["goal"])
            grads = torch.autograd.grad(
                q_value, x, grad_outputs=torch.ones_like(q_value)
            )[0]
            grads = grads.detach()

        return out + cond_lambda * grads * (sigma**2).view(-1, 1, 1)

    def get_params(self):
        return self.model.get_params()
