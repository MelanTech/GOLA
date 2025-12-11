import random
from itertools import combinations

import torch
import math
import torch.nn as nn
from timm.layers import trunc_normal_

from ..lora import LoRALayer
from .merge import _lora_merge, _lora_unmerge


class GOLALayer(LoRALayer):
    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, dropout: float,
                 rs_lora: bool = False, init_method: str = 'bert', r_retained: int = 16, n_groups: int = 8):
        super().__init__(in_dim, out_dim, r, alpha, dropout, rs_lora, init_method)
        self.A = nn.Parameter(torch.empty(r_retained, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, r_retained))
        self.GA = nn.ParameterList(
            [nn.Parameter(torch.empty((r - r_retained) // n_groups, in_dim)) for _ in range(n_groups)]
        )
        self.GB = nn.ParameterList(
            [nn.Parameter(torch.empty(out_dim, (r - r_retained) // n_groups)) for _ in range(n_groups)]
        )

        if init_method == 'lora':
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif init_method == 'gaussian':
            nn.init.normal_(self.A, std=1. / self.r)
            nn.init.zeros_(self.B)
        elif init_method == 'bert':
            trunc_normal_(self.A, std=.02)
            trunc_normal_(self.B, std=.02)
        else:
            raise ValueError(f'Unknown init method: {init_method}')

    def forward(self, x: torch.Tensor):
        A = torch.cat([self.A, *self.GA], dim=0)
        B = torch.cat([self.B, *self.GB], dim=1)
        return (self.dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling


class LinearWithGOLA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, rs_lora: bool = False,
                 init_method: str = 'bert', merge_on_eval: bool = False):
        super().__init__()
        self.linear = linear
        self.lora = GOLALayer(linear.in_features, linear.out_features, r, alpha, dropout, rs_lora, init_method)
        self.merge_on_eval = merge_on_eval
        self.merged = False

    def forward(self, x: torch.Tensor):
        if self.merged:
            return self.linear(x)
        else:
            return self.linear(x) + self.lora(x)

    def train(self, mode: bool = True):
        if self.merge_on_eval:
            if not mode:
                self.merge()
            else:
                self.unmerge()

        return super().train(mode)

    def state_dict(self, **kwargs):
        merged = self.merged
        if merged:
            self.unmerge()
        state_dict = super().state_dict(**kwargs)
        if merged:
            self.merge()
        return state_dict

    def load_state_dict(self, **kwargs):
        merged = self.merged
        if merged:
            self.unmerge()
        ret = super().load_state_dict(**kwargs)
        if merged:
            self.merge()
        return ret

    def merge(self):
        if self.merged:
            return
        self.linear.weight.data.copy_(
            _lora_merge(self.linear.weight.data, torch.cat([self.lora.A, *self.lora.GA], dim=0),
                        torch.cat([self.lora.B, *self.lora.GB], dim=1), self.lora.alpha, self.lora.rs_lora))
        self.merged = True

    def unmerge(self):
        if not self.merged:
            return
        self.linear.weight.data.copy_(
            _lora_unmerge(self.linear.weight.data, torch.cat([self.lora.A, *self.lora.GA], dim=0),
                          torch.cat([self.lora.B, *self.lora.GB], dim=1), self.lora.alpha, self.lora.rs_lora))
        self.merged = False


class LinearWithGOLA_TimmQKV(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, rs_lora: bool = False,
                 init_method: str = 'bert',
                 target_q: bool = True, target_k: bool = True, target_v: bool = True):
        super().__init__()
        dim = linear.in_features
        bias = linear.bias is not None
        q = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        k = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=linear.weight.device, dtype=linear.weight.dtype)
        q.weight.data.copy_(linear.weight.data[:dim])
        k.weight.data.copy_(linear.weight.data[dim:2 * dim])
        v.weight.data.copy_(linear.weight.data[2 * dim:])
        q.weight.requires_grad = k.weight.requires_grad = v.weight.requires_grad = linear.weight.requires_grad
        if bias:
            q.bias.data.copy_(linear.bias.data[:dim])
            k.bias.data.copy_(linear.bias.data[dim:2 * dim])
            v.bias.data.copy_(linear.bias.data[2 * dim:])
            q.bias.requires_grad = k.bias.requires_grad = v.bias.requires_grad = linear.bias.requires_grad

        if target_q:
            self.q = LinearWithGOLA(q, r, alpha, dropout, rs_lora, init_method)
        else:
            self.q = q
        if target_k:
            self.k = LinearWithGOLA(k, r, alpha, dropout, rs_lora, init_method)
        else:
            self.k = k
        if target_v:
            self.v = LinearWithGOLA(v, r, alpha, dropout, rs_lora, init_method)
        else:
            self.v = v

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat((q, k, v), dim=-1)


def compute_rand_pair_orth_loss(model: nn.Module, weight: float = 1.0, n_groups: int = 4):
    loss = 0
    param_dict = dict(model.named_parameters())
    param_counter = 0

    for n in param_dict.keys():
        if "lora.GA.0" in n:
            prefix = n[:-len("lora.GA.0")]
            w = [param_dict[prefix + f"lora.GA.{i}"] for i in range(n_groups)]

            base_matrix = random.sample(w, 2)
            loss_comb = weight * torch.sum(torch.mm(base_matrix[0].t(), base_matrix[1]).abs())

            loss += loss_comb if n_groups > 1 else 0
            param_counter += 1

        elif "lora.GB.0" in n:
            prefix = n[:-len("lora.GB.0")]
            w = [param_dict[prefix + f"lora.GB.{i}"] for i in range(n_groups)]

            base_matrix = random.sample(w, 2)
            loss_comb = weight * torch.sum(torch.mm(base_matrix[0].t(), base_matrix[1]).abs())

            loss += loss_comb if n_groups > 1 else 0
            param_counter += 1

    return loss / param_counter
