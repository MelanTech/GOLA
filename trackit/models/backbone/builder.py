import copy
from datetime import timedelta
from trackit.core.runtime.global_constant import get_global_constant
from trackit.miscellanies.torch.distributed.barrier import torch_distributed_zero_first


def _build_backbone(backbone_config: dict, load_pretrained=True, torch_jit_trace_compatible=True):
    backbone_config = copy.deepcopy(backbone_config)
    if 'parameters' in backbone_config:
        backbone_build_params = backbone_config['parameters']
        if load_pretrained and 'pretrained' in backbone_build_params:
            load_pretrained = backbone_build_params['pretrained']
            del backbone_build_params['pretrained']
    else:
        backbone_build_params = {}

    if backbone_config['type'] == 'DINOv2':
        from .dinov2.builder import build_dino_v2_backbone
        if torch_jit_trace_compatible:
            backbone_build_params['acc'] = 'none'
        backbone = build_dino_v2_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone


def build_backbone(backbone_config: dict, load_pretrained=True, torch_jit_trace_compatible=False):
    with torch_distributed_zero_first(on_local_master=not get_global_constant('on_shared_file_system'), timeout=timedelta(hours=5)):
        return _build_backbone(backbone_config, load_pretrained, torch_jit_trace_compatible)
