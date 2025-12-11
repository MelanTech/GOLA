# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add simple template updater

from typing import Tuple
import torch
import numpy as np

from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping
from . import TemplateUpdater
from ..tensor_cache import CacheService, TensorCache


class SimpleTemplateUpdater(TemplateUpdater):
    def __init__(self, threshold: float, template_area_factor: float, template_size: Tuple[int, int],
                 norm_stats_dataset_name, interpolation_mode, interpolation_align_corners, device: torch.device):
        super().__init__()
        assert 0 <= threshold <= 1
        assert template_area_factor > 0

        self.threshold = threshold
        self.template_area_factor = template_area_factor

        self.template_size = np.array(template_size)
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

        self.transforms = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)

        self.template_cache = None

        self.device = device

    def start(self, max_batch_size: int, template_shape: Tuple[int, int, int]):
        self.template_cache = CacheService(max_batch_size,
                                           TensorCache(max_batch_size, template_shape, self.device))

    def stop(self):
        del self.template_cache
        self.template_cache = None

    def initialize(self, task_id, template: torch.Tensor):
        self.template_cache.put(task_id, template)

    def update(self, task_id, confidence: float, x: torch.Tensor, bbox: np.ndarray):
        if confidence > self.threshold:
            d = self._crop_template(x, bbox)
            self.template_cache.update(task_id, d)

    def delete(self, task_id):
        self.template_cache.delete(task_id)

    def get(self, task_id) -> torch.Tensor:
        return self.template_cache.get(task_id)

    def get_batch(self, task_ids: list) -> torch.Tensor:
        return self.template_cache.get_batch(task_ids)

    def _crop_template(self, x: torch.Tensor, bbox) -> torch.Tensor:
        template_curation_parameter = get_siamfc_cropping_params(bbox, self.template_area_factor,
                                                                 self.template_size)

        d, _, _ = apply_siamfc_cropping(
            x.to(torch.float32), self.template_size, template_curation_parameter,
            self.interpolation_mode, self.interpolation_align_corners)

        return self.transforms(d.div_(255.))
