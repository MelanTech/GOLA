# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add build support for online template

import torch

from trackit.miscellanies.pretty_format import pretty_format
from . import TemplateUpdater


def build_template_updater(template_update_config: dict, common_config: dict, device: torch.device) -> TemplateUpdater:
    print('Tracker template update:\n' + pretty_format(template_update_config))
    template_update_type = template_update_config['type']
    if template_update_type == 'simple':
        from .simple import SimpleTemplateUpdater
        return SimpleTemplateUpdater(threshold=template_update_config['update_threshold'],
                                     template_area_factor=template_update_config['template_area_factor'],
                                     template_size=common_config['template_size'],
                                     norm_stats_dataset_name=common_config['normalization'],
                                     interpolation_mode=common_config['interpolation_mode'],
                                     interpolation_align_corners=common_config['interpolation_align_corners'],
                                     device=device)
    else:
        raise NotImplementedError("Unknown template update type: {}".format(template_update_type))
