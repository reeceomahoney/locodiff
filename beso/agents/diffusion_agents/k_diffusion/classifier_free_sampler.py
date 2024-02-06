import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from . import utils


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
# code adapted from 
# https://github.com/GuyTevet/motion-diffusion-model/blob/55cd84e14ba8e0e765700913023259b244f5e4ed/model/cfg_sampler.py
class ClassifierFreeSampleModel(nn.Module):
    """
    A wrapper model that adds conditional sampling capabilities to an existing model.

    Args:
        model (nn.Module): The underlying model to run.
        cond_lambda (float): Optional. The conditional lambda value. Defaults to 2.

    Attributes:
        model (nn.Module): The underlying model.
        cond_lambda (float): The conditional lambda value.
        cond (bool): Indicates whether conditional sampling is enabled based on the cond_lambda value.
    """
    def __init__(self, model, cond_lambda: float):
        super().__init__()
        # TODO: refactor this into beso agent
        self.model = model
        self.cond_lambda = cond_lambda

    def forward(self, state_action, goal, sigma, **extra_args):
        state_action = deepcopy(state_action)

        # unconditional output
        uncond_dict = {'uncond': True}
        out_uncond = self.model(state_action, goal, sigma, **uncond_dict)

        # (n_envs * n_cond, t, D)
        if torch.sum(goal) > 0:
            diag = torch.eye(goal.shape[-1]).to(goal.device)
            diag = diag.unsqueeze(0).repeat(goal.shape[0], 1, 1)
            diag = diag.reshape(-1, diag.shape[-1])

            state_action_repeat = state_action.unsqueeze(1).repeat(1, goal.shape[-1], 1, 1)
            state_action_repeat = state_action_repeat.reshape(-1, *state_action.shape[1:])

            sigma = sigma.repeat(goal.shape[-1])
            out = self.model(state_action_repeat, diag.unsqueeze(1), sigma)
            out = out.reshape(goal.shape[0], goal.shape[-1], *out.shape[1:])

            mask = goal.squeeze().bool().unsqueeze(-1).unsqueeze(-1)
            out_uncond_ = out_uncond.unsqueeze(1).repeat(1, goal.shape[-1], 1, 1)
            out_neg = ((out - out_uncond_) * mask).sum(dim=1)

            out_uncond -= self.cond_lambda * out_neg
            
        return out_uncond
    
    def get_params(self):
        return self.model.get_params()
    


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
    def __init__(self, model, cond_func, cond_lambda: float=2):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.guide = cond_func
        # pointers to inner model
        self.cond_lambda = cond_lambda
        
    def forward(self, state, action, goal, sigma, cond_lambda=None, **extra_args):
        if cond_lambda is None:
            cond_lambda = self.cond_lambda
        pred_action = self.model(state, action, goal, sigma, **extra_args)
        with torch.enable_grad():
            a = pred_action.clone().requires_grad_(True)
            q_value = self.guide(state, pred_action, goal)
            grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
        
        return pred_action + cond_lambda * grads * utils.append_dims(sigma**2, action.ndim)
        
    def get_params(self):
        return self.model.get_params()