# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for RGB-T data visualization

import numpy as np
import torch
import os
import cv2
from typing import Optional
from trackit.core.runtime.context.task import get_current_task_context
from trackit.miscellanies.image.io import write_image
from trackit.miscellanies.image.draw import draw_box_on_image_


def visualize_tracking_result(dataset_name: str, track_name: str, frame_index: int,
                              template: torch.Tensor,
                              search_region: torch.Tensor,
                              online_template: torch.Tensor,
                              predicted_box: np.ndarray,
                              predicted_mask: Optional[np.ndarray],
                              predicted_mask_on_full_search_image: Optional[np.ndarray]):
    output_path = get_current_task_context().get_current_epoch_output_path()
    if output_path is None:
        return
    output_path = os.path.join(output_path, 'tracking_result_visualization', track_name)
    os.makedirs(output_path, exist_ok=True)
    output_image_path = os.path.join(output_path, f'{dataset_name}_{frame_index}.png')

    template = denormalize_image(template)
    online_template = denormalize_image(online_template)

    search_region = search_region.permute(1, 2, 0).to(torch.uint8).cpu().numpy(force=True)
    template = template.to(torch.uint8).cpu().numpy(force=True)
    online_template = online_template.to(torch.uint8).cpu().numpy(force=True)

    search_region = search_region[:, :, :3]
    template = template[:, :, :3]
    online_template = online_template[:, :, :3]

    if predicted_box is not None:
        draw_box_on_image_(search_region, predicted_box, color=(0, 255, 0), thickness=2)

    output_image = np.hstack([search_region, np.vstack([template, online_template])])

    write_image(output_image, output_image_path)

    # if predicted_mask is not None:
    #     output_image_path = os.path.join(output_path, f'{dataset_name}_{track_name}_{frame_index}_x_mask.png')
    #     write_image(predicted_mask, output_image_path)
    # if predicted_mask_on_full_search_image is not None:
    #     output_image_path = os.path.join(output_path, f'{dataset_name}_{track_name}_{frame_index}_x_mask_full.png')
    #     write_image(predicted_mask_on_full_search_image, output_image_path)


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406, 0.449, 0.449, 0.449],
                      std=[0.229, 0.224, 0.225, 0.226, 0.226, 0.226]):
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)
    image_tensor = image_tensor.permute(1, 2, 0)
    denormalized_image = image_tensor * std_tensor + mean_tensor

    return denormalized_image * 255
