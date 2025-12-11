# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for online template feature mask generation

from typing import Tuple, Sequence, Mapping
import numpy as np
import torch

from trackit.data.protocol.train_input import TrainData
from trackit.data.utils.collation_helper import collate_element_as_torch_tensor
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize


class TemplateFeatMaskGenerator:
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int]):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = (
            self.template_size[0] / self.template_feat_size[0], self.template_size[1] / self.template_feat_size[1])
        self.background_value = 0
        self.foreground_value = 1

    def __call__(self, training_pair, context: dict, data: dict, _):
        z_cropped_bbox = context['z_cropped_bbox']
        d_cropped_bbox = context['d_cropped_bbox']

        z_mask = self._generate_mask(z_cropped_bbox)
        d_mask = self._generate_mask(d_cropped_bbox)

        data['z_cropped_bbox_feat_map_mask'] = z_mask
        data['d_cropped_bbox_feat_map_mask'] = d_mask

    def _generate_mask(self, z_cropped_bbox: np.ndarray):
        mask = np.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value, dtype=np.int64)
        z_cropped_bbox = z_cropped_bbox.copy()
        z_cropped_bbox[0] /= self.stride[0]
        z_cropped_bbox[1] /= self.stride[1]
        z_cropped_bbox[2] /= self.stride[0]
        z_cropped_bbox[3] /= self.stride[1]
        z_cropped_bbox = bbox_rasterize(z_cropped_bbox, dtype=np.int64)
        mask[z_cropped_bbox[1]:z_cropped_bbox[3], z_cropped_bbox[0]:z_cropped_bbox[2]] = self.foreground_value
        return mask


def template_feat_mask_data_collator(batch: Sequence[Mapping], collated: TrainData):
    collated.input['z_feat_mask'] = collate_element_as_torch_tensor(batch, 'z_cropped_bbox_feat_map_mask')
    collated.input['d_feat_mask'] = collate_element_as_torch_tensor(batch, 'd_cropped_bbox_feat_map_mask')
