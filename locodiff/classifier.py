import logging

import torch
from torch import nn

from .utils import SinusoidalPosEmb

log = logging.getLogger(__name__)


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
        obs_dim,
        skill_dim,
        act_dim,
        d_model,
        nhead,
        num_layers,
        T,
        T_cond,
        device,
        dropout,
        weight_decay: float,
        ddpm: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device
        self.weight_decay = weight_decay
        self.ddpm = ddpm

        self.action_emb = nn.Linear(self.obs_dim + self.act_dim, self.d_model)
        self.obs_emb = nn.Linear(self.obs_dim + 1, self.d_model)

        self.drop = nn.Dropout(dropout)

        self.pos_emb = (
            SinusoidalPosEmb(d_model)(torch.arange(T)).unsqueeze(0).to(device)
        )
        self.cond_pos_emb = (
            SinusoidalPosEmb(d_model)(torch.arange(T_cond)).unsqueeze(0).to(device)
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=4 * self.d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.num_layers,
        )
        mask = self.generate_mask(T)
        self.register_buffer("mask", mask)

        self.ln_f = nn.LayerNorm(self.d_model)
        self.action_pred = nn.Linear(d_model, 1)

        self.apply(self._init_weights)
        self.to(device)

        total_params = sum(p.numel() for p in self.get_params())
        log.info(f"Total parameters: {total_params:e}")

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerDecoderLayer,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.SiLU,
            nn.Sequential,
            ClassifierTransformer,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, noised_action, data_dict):
        # embeddings
        obs = data_dict["obs"]
        vel_cmd = data_dict["vel_cmd"].unsqueeze(1).expand(-1, obs.shape[1], -1)
        obs = torch.cat([obs, vel_cmd], dim=-1)
        obs_emb = self.obs_emb(obs)
        action_emb = self.action_emb(noised_action)

        cond = obs_emb
        cond += self.cond_pos_emb
        cond = self.drop(cond)

        action_emb += self.pos_emb
        x = self.decoder(tgt=action_emb, memory=cond, tgt_mask=self.mask)
        x = self.ln_f(x)
        out = self.action_pred(x)
        out = self.drop(out)

        return out

    def generate_mask(self, x):
        mask = (torch.triu(torch.ones(x, x)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def get_params(self):
        return self.parameters()

    def detach_all(self):
        for _, param in self.named_parameters():
            param.detach_()


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

    def __init__(self, model, cond_func, cond_lambda):
        super().__init__()
        self.model = model
        self.guide = cond_func
        self.cond_lambda = cond_lambda

    def forward(self, noised_action, sigma, data_dict):
        out = self.model(noised_action, sigma, data_dict)
        with torch.enable_grad():
            x = out.clone().requires_grad_(True)
            q_value = self.guide(x, data_dict)
            grads = torch.autograd.grad(
                q_value, x, grad_outputs=torch.ones_like(q_value)
            )[0]
            grads = grads.detach()

        return out + self.cond_lambda * grads * (sigma**2).view(-1, 1, 1)

    def get_params(self):
        return self.model.get_params()
