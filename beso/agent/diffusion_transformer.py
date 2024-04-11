import torch
import torch.nn as nn


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim,
        pred_obs_dim,
        act_dim,
        d_model,
        nhead,
        num_layers,
        T,
        T_cond,
        device,
        cond_mask_prob,
        dropout,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.pred_obs_dim = pred_obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.T = T
        self.T_cond = T_cond
        self.device = device
        self.cond_mask_prob = cond_mask_prob

        self.state_action_emb = nn.Linear(
            self.pred_obs_dim + self.act_dim, self.d_model
        )
        self.cond_state_emb = nn.Linear(self.obs_dim, self.d_model)
        self.sigma_emb = nn.Linear(1, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.T + 1, d_model))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, self.T_cond + 1, d_model))

        self.encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model)
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
        mask = self.generate_mask(T + 1)
        self.register_buffer("mask", mask)

        self.ln_f = nn.LayerNorm(self.d_model)
        self.state_action_pred = nn.Linear(d_model, self.pred_obs_dim + self.act_dim)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerDecoderLayer,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
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
        elif isinstance(module, DiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
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
            for pn, p in m.named_parameters():
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

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

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
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, x, cond, sigma, **kwargs):
        """
        x: [batch_size, T, pred_obs_dim + act_dim] noise vector
        cond: [batch_size, T_cond, obs_dim] observation history
        sigma: [batch_size] noise level
        """
        cond_emb = self.cond_state_emb(cond)
        input_emb = self.state_action_emb(x)

        # diffusion timestep embedding
        sigma = sigma.view(-1, 1, 1).log() / 4
        sigma_emb = self.sigma_emb(sigma)

        cond = torch.cat([sigma_emb, cond_emb], dim=1)
        cond += self.cond_pos_emb
        cond = self.encoder(cond)

        input_emb += self.pos_emb
        x = self.decoder(
            tgt=input_emb,
            memory=cond,
            tgt_mask=self.mask,
        )
        x = self.ln_f(x)
        return self.state_action_pred(x)

    def generate_mask(self, x):
        mask = (torch.triu(torch.ones(x, x)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def mask_cond(self, cond, force_mask=False):
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0:
            mask = torch.rand_like(cond[:, :, 0]) < self.cond_mask_prob
            return cond * mask.unsqueeze(-1).float()
        else:
            return cond

    def get_params(self):
        return self.parameters()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTransformer(
        obs_dim=33,
        act_dim=12,
        d_model=128,
        nhead=4,
        num_layers=4,
        T=6,
        T_cond=4,
        device=device,
    )
    x = torch.randn(1, 6, 45).to(device)
    cond = torch.randn(1, 4, 45).to(device)
    goal = torch.randn(1, 1).to(device)
    sigma = torch.tensor([1]).to(device)
    model(x, cond, sigma)
