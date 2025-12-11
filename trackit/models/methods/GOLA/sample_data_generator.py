# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for RGB-T data

import torch
from typing import Tuple

from trackit.models import SampleInputDataGeneratorInterface


class GOLA_DummyDataGenerator(SampleInputDataGeneratorInterface):
    def __init__(self, template_size: Tuple[int, int], search_region_size: Tuple[int, int], template_feat_map_size: Tuple[int, int]):
        self._template_size = template_size
        self._search_region_size = search_region_size
        self._template_feat_map_size = template_feat_map_size

    def get(self, batch_size: int, device: torch.device):
        return {'z': torch.full((batch_size, 6, self._template_size[1], self._template_size[0]), 0.5, device=device),
                'x': torch.full((batch_size, 6, self._search_region_size[1], self._search_region_size[0]), 0.5, device=device),
                'd': torch.full((batch_size, 6, self._template_size[1], self._template_size[0]), 0.5, device=device),
                'z_feat_mask': torch.full((batch_size, self._template_feat_map_size[1], self._template_feat_map_size[0]), 1, dtype=torch.long, device=device),
                'd_feat_mask': torch.full((batch_size, self._template_feat_map_size[1], self._template_feat_map_size[0]), 1, dtype=torch.long, device=device)
                }


def build_sample_input_data_generator(config: dict):
    common_config = config['common']
    return GOLA_DummyDataGenerator(common_config['template_size'], common_config['search_region_size'], common_config['template_feat_size'])
