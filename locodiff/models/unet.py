import logging
from functools import partial

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from locodiff.utils import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        obs_dim,
        input_dim,
        T_cond,
        cond_embed_dim,
        down_dims,
        device,
        cond_mask_prob,
        weight_decay: float,
        local_cond_dim=None,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # diffusion step embedding, observations, vel_cmd, return
        cond_dim = cond_embed_dim + (obs_dim * T_cond) + 4

        CondResBlock = partial(
            ConditionalResidualBlock1D,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [CondResBlock(dim_in, dim_out), CondResBlock(dim_in, dim_out)]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [CondResBlock(mid_dim, mid_dim), CondResBlock(mid_dim, mid_dim)]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        CondResBlock(dim_in, dim_out),
                        CondResBlock(dim_out, dim_out),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        CondResBlock(dim_out * 2, dim_in),
                        CondResBlock(dim_in, dim_in),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        # self.obs_emb = nn.Linear(obs_dim, cond_embed_dim)
        # self.vel_cmd_emb = nn.Linear(3, cond_embed_dim)
        # self.return_emb = nn.Linear(1, cond_embed_dim)

        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(cond_embed_dim),
            nn.Linear(cond_embed_dim, cond_embed_dim * 4),
            nn.Mish(),
            nn.Linear(cond_embed_dim * 4, cond_embed_dim),
        )

        self.cond_mask_prob = cond_mask_prob
        self.weight_decay = weight_decay

        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        self.to(device)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        noised_action: torch.Tensor,
        sigma: torch.Tensor,
        data_dict: dict,
        uncond: bool = False,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(noised_action, "b t h -> b h t")

        obs = data_dict["obs"].reshape(sample.shape[0], -1)
        vel_cmd = data_dict["vel_cmd"]
        returns = self.mask_cond(data_dict["return"], uncond)
        sigma_emb = self.sigma_encoder(sigma)

        global_feature = torch.cat([sigma_emb, returns, vel_cmd, obs], dim=-1)

        # encode local features
        h_local = list()
        local_cond = None
        if self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, "b t h -> b h t")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == (len(self.up_modules)) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b h t -> b t h")
        return x

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

    def get_optim_groups(self):
        return [{"params": self.parameters(), "weight_decay": self.weight_decay}]

    def get_params(self):
        return self.parameters()
