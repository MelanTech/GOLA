import os.path
import torch
import torchvision
import numpy as np
from typing import Optional
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform_reversed

from trackit.data.methods.siamese_tracker_train._types import SOTFrameInfo
from trackit.data.methods.siamese_tracker_train.worker import SiameseTrainingPair


def _get_original_image(image: torch.Tensor, bbox: Optional[np.ndarray],
                        dataset_norm_stats: Optional[str]) -> torch.Tensor:
    assert image.dtype == torch.float
    if dataset_norm_stats is not None:
        image = get_dataset_norm_stats_transform_reversed(dataset_norm_stats, inplace=False)(image)
        image.mul_(255.0)
        image.clamp_(min=0, max=255.)
    image = image.to(torch.uint8)
    if bbox is not None:
        if image.shape[0] == 6:
            image_v = torchvision.utils.draw_bounding_boxes(image[:3], torch.from_numpy(bbox).unsqueeze(0),
                                                            width=2, colors='red')
            image_i = torchvision.utils.draw_bounding_boxes(image[3:], torch.from_numpy(bbox).unsqueeze(0),
                                                            width=2, colors='red')
            image = torch.cat([image_v, image_i], dim=0)
        else:
            image = torchvision.utils.draw_bounding_boxes(image, torch.from_numpy(bbox).unsqueeze(0), width=2)
    return image


def visualize_element(name: str, context: dict, element_info: SOTFrameInfo, output_path: str,
                      dataset_norm_stats: Optional[str] = None):
    file_name = f'{element_info.track.get_name()}_{element_info.frame.get_frame_index()}_{name}.jpg'
    image = context[f'{name}_image']
    bbox = None
    if f'{name}_bbox' in context:
        bbox = context[f'{name}_bbox']
    image = _get_original_image(image, bbox, dataset_norm_stats)
    if image.shape[0] == 6:
        base, ext = os.path.splitext(file_name)
        file_name_v = f"{base}_v{ext}"
        file_name_i = f"{base}_i{ext}"
        torchvision.io.write_jpeg(image[:3], os.path.join(output_path, file_name_v))
        torchvision.io.write_jpeg(image[3:], os.path.join(output_path, file_name_i))
    else:
        torchvision.io.write_jpeg(image, os.path.join(output_path, file_name))


def visualize_siam_tracker_training_pair_processor(output_path: str, training_pair: SiameseTrainingPair, context: dict,
                                                   dataset_norm_stats: str):
    if 'z_image' in context:
        visualize_element('z', context, training_pair.template, output_path)
    if 'x_image' in context:
        visualize_element('x', context, training_pair.search, output_path)
    if 'z_cropped_image' in context:
        visualize_element('z_cropped', context, training_pair.template, output_path, dataset_norm_stats)
    if 'x_cropped_image' in context:
        visualize_element('x_cropped', context, training_pair.search, output_path, dataset_norm_stats)
    if 'd_image' in context:
        visualize_element('d', context, training_pair.online, output_path)
    if 'd_cropped_image' in context:
        visualize_element('d_cropped', context, training_pair.online, output_path, dataset_norm_stats)
