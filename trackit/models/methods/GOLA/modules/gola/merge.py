import math
from typing import Mapping, Any, Optional
from collections import OrderedDict
import torch
import torch.nn as nn


def lora_merge_state_dict(module: nn.Module, state_dict: Mapping[str, Any]) -> OrderedDict:
    state_dict = OrderedDict(**state_dict)
    lora_alpha = None
    use_rslora = False
    if 'lora_alpha' in state_dict:
        lora_alpha = state_dict['lora_alpha'].item()
        use_rslora = state_dict['use_rslora'].item()
        del state_dict['lora_alpha']
        del state_dict['use_rslora']
    for name in list(state_dict.keys()):
        if 'q.lora.A' in name:
            device = state_dict[name].device
            prefix = name[:-len('.q.lora.A')]
            qkv_module: nn.Linear = module.get_submodule(prefix)
            state_dict_has_linear_weight = prefix + '.q.lora.linear.weight' in state_dict
            state_dict_has_linear_bias = prefix + '.q.lora.linear.bias' in state_dict
            dim = qkv_module.in_features

            q_A = state_dict[prefix + '.q.lora.A']
            q_B = state_dict[prefix + '.q.lora.B']
            q_GA = _cat_grouped_param(state_dict, prefix, '.q.lora.GA', n_groups=8, dim=0)
            q_GB = _cat_grouped_param(state_dict, prefix, '.q.lora.GB', n_groups=8, dim=1)
            q_A = torch.cat((q_A, q_GA), dim=0)
            q_B = torch.cat((q_B, q_GB), dim=1)
            if state_dict_has_linear_weight:
                q_linear_weight = state_dict[prefix + '.q.lora.linear.weight']
            else:
                q_linear_weight = qkv_module.weight.data[:dim].to(device)
            q_merged_weight = _lora_merge(q_linear_weight, q_A, q_B, lora_alpha, use_rslora)
            has_lora_k = (prefix + '.k.lora.A') in state_dict
            if has_lora_k:
                k_A = state_dict[prefix + '.k.lora.A']
                k_B = state_dict[prefix + '.k.lora.B']
                k_GA = _cat_grouped_param(state_dict, prefix, '.k.lora.GA', n_groups=8, dim=0)
                k_GB = _cat_grouped_param(state_dict, prefix, '.k.lora.GB', n_groups=8, dim=1)
                k_A = torch.cat((k_A, k_GA), dim=0)
                k_B = torch.cat((k_B, k_GB), dim=1)
                if state_dict_has_linear_weight:
                    k_linear_weight = state_dict[prefix + '.k.lora.linear.weight']
                else:
                    k_linear_weight = qkv_module.weight.data[dim:2 * dim].to(device)
                k_merged_weight = _lora_merge(k_linear_weight, k_A, k_B, lora_alpha, use_rslora)
            else:
                k_linear_weight = qkv_module.weight.data[dim:2 * dim].to(device)
                k_merged_weight = k_linear_weight
            v_A = state_dict[prefix + '.v.lora.A']
            v_B = state_dict[prefix + '.v.lora.B']
            v_GA = _cat_grouped_param(state_dict, prefix, '.v.lora.GA', n_groups=8, dim=0)
            v_GB = _cat_grouped_param(state_dict, prefix, '.v.lora.GB', n_groups=8, dim=1)
            v_A = torch.cat((v_A, v_GA), dim=0)
            v_B = torch.cat((v_B, v_GB), dim=1)
            if state_dict_has_linear_weight:
                v_linear_weight = state_dict[prefix + '.v.lora.linear.weight']
            else:
                v_linear_weight = qkv_module.weight.data[2 * dim:].to(device)
            v_merged_weight = _lora_merge(v_linear_weight, v_A, v_B, lora_alpha, use_rslora)
            qkv_merged_weight = torch.cat((q_merged_weight, k_merged_weight, v_merged_weight), dim=0)
            state_dict[prefix + '.weight'] = qkv_merged_weight

            if state_dict_has_linear_bias:
                q_bias = state_dict[prefix + '.q.lora.linear.bias']
                k_bias = state_dict[prefix + '.k.lora.linear.bias']
                v_bias = state_dict[prefix + '.v.lora.linear.bias']
                qkv_merged_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                state_dict[prefix + '.bias'] = qkv_merged_bias

            if state_dict_has_linear_weight:
                del state_dict[prefix + '.q.lora.linear.weight']
                del state_dict[prefix + '.k.lora.linear.weight']
                del state_dict[prefix + '.v.lora.linear.weight']
            if state_dict_has_linear_bias:
                del state_dict[prefix + '.q.lora.linear.bias']
                del state_dict[prefix + '.k.lora.linear.bias']
                del state_dict[prefix + '.v.lora.linear.bias']

            del state_dict[prefix + '.q.lora.A']
            del state_dict[prefix + '.q.lora.B']
            _del_grouped_param(state_dict, prefix, '.q.lora.GA', n_groups=8)
            _del_grouped_param(state_dict, prefix, '.q.lora.GB', n_groups=8)
            if has_lora_k:
                del state_dict[prefix + '.k.lora.A']
                del state_dict[prefix + '.k.lora.B']
                _del_grouped_param(state_dict, prefix, '.k.lora.GA', n_groups=8)
                _del_grouped_param(state_dict, prefix, '.k.lora.GB', n_groups=8)
            del state_dict[prefix + '.v.lora.A']
            del state_dict[prefix + '.v.lora.B']
            _del_grouped_param(state_dict, prefix, '.v.lora.GA', n_groups=8)
            _del_grouped_param(state_dict, prefix, '.v.lora.GB', n_groups=8)
    for name in list(state_dict.keys()):
        if 'lora.A' in name:
            device = state_dict[name].device
            prefix = name[:-len('.lora.A')]
            state_dict_has_linear_weight = prefix + '.linear.weight' in state_dict
            state_dict_has_linear_bias = prefix + '.linear.bias' in state_dict
            if state_dict_has_linear_weight:
                linear_weight = state_dict[prefix + '.linear.weight']
            else:
                linear_weight = module.get_submodule(prefix).weight.data.to(device)
            A = state_dict[prefix + '.lora.A']
            B = state_dict[prefix + '.lora.B']
            GA = _cat_grouped_param(state_dict, prefix, '.lora.GA', n_groups=8, dim=0)
            GB = _cat_grouped_param(state_dict, prefix, '.lora.GB', n_groups=8, dim=1)
            A = torch.cat((A, GA), dim=0)
            B = torch.cat((B, GB), dim=1)
            merged_weight = _lora_merge(linear_weight, A, B, lora_alpha, use_rslora)
            state_dict[prefix + '.weight'] = merged_weight
            if state_dict_has_linear_bias:
                state_dict[prefix + '.bias'] = state_dict[prefix + '.linear.bias']
            if state_dict_has_linear_weight:
                del state_dict[prefix + '.linear.weight']
            if state_dict_has_linear_bias:
                del state_dict[prefix + '.linear.bias']
            del state_dict[prefix + '.lora.A']
            del state_dict[prefix + '.lora.B']
            _del_grouped_param(state_dict, prefix, '.lora.GA', n_groups=8)
            _del_grouped_param(state_dict, prefix, '.lora.GB', n_groups=8)

    return state_dict


def _lora_delta(lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float], use_rslora: bool) -> torch.Tensor:
    r = lora_A.size(0)
    if alpha is not None:
        if use_rslora:
            scaling = alpha / math.sqrt(r)
        else:
            scaling = alpha / r
    else:
        scaling = 1.
    return (lora_B @ lora_A) * scaling


def _lora_merge(weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float],
                use_rslora: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _lora_delta(lora_A.to(torch.float32), lora_B.to(torch.float32), alpha, use_rslora)

    return (weight.to(torch.float32) + delta).to(original_dtype)


def _lora_unmerge(weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, alpha: Optional[float],
                  use_rslora: bool) -> torch.Tensor:
    original_dtype = weight.dtype

    delta = _lora_delta(lora_A.to(torch.float32), lora_B.to(torch.float32), alpha, use_rslora)

    return (weight.to(torch.float32) - delta).to(original_dtype)


def _cat_grouped_param(state_dict: OrderedDict, prefix: str, name: str, n_groups: int, dim=0):
    params = [state_dict[f'{prefix}{name}.{i}'] for i in range(n_groups)]
    return torch.cat(params, dim=dim)


def _del_grouped_param(state_dict: OrderedDict, prefix: str, name: str, n_groups: int):
    for i in range(n_groups):
        del state_dict[f'{prefix}{name}.{i}']
