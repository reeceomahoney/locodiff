import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops
import hydra

logger = logging.getLogger(__name__)

# Code from miniGPT adopted for score-based diffusion models 
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class DiffusionGPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        state_dim: int,
        device: str,
        action_dim: int,
        goal_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        obs_seq_len: int,
        future_seq_len: int,
        goal_drop: float = 0,
    ):
        super().__init__()
        self.device = device

        # input embedding stem (2 for the sigma and the goal)
        self.future_seq_len = future_seq_len
        self.horizon = obs_seq_len + future_seq_len
        block_size = 2 + self.horizon
        self.tok_emb = nn.Linear(state_dim + action_dim, embed_dim)
        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        
        # needed for classifier guidance learning
        self.cond_mask_prob = goal_drop
        
        # transformer
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size
            ) for _ in range(n_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.block_size = block_size
        self.sigma_emb = nn.Linear(1, embed_dim)

        # pred module 
        self.state_action_pred = nn.Linear(embed_dim, state_dim + action_dim)
        
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DiffusionGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

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
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward( self, state_actions, goals, sigma, uncond: Optional[bool] =False):  
        t = state_actions.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the sigma embedding
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas.to(torch.float32))
        if len(state_actions.shape) == 3 and len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
              
        # during training randomly mask out the goal for CFG
        if self.training:
            goals = self.mask_cond(goals)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  
        
        # embed them into linear representations for the transformer
        state_action_embed = self.tok_emb(state_actions)
        goal_embed = self.goal_emb(goals)
        
        position_embeddings = self.pos_emb[ :, :(t + 1), : ]

        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        goal_x = self.drop(goal_embed + position_embeddings[:, :1, :])
        sa_x = self.drop(state_action_embed + position_embeddings[:, 1:, :])
        input_seq = torch.cat([emb_t, goal_x, sa_x], dim=1)
        
        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # remove the sigma and goal embeddings
        x = x[:, 2:, :]

        pred_state_action = self.state_action_pred(x)

        return pred_state_action
    
    def mask_cond(self, cond, force_mask=False):
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((cond.shape[0], 1, 1), device=cond.device) * self.cond_mask_prob)
            mask = mask.expand_as(cond)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()
