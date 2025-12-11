# Modified by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for RGB-T datasets

import os.path

import numpy as np
import torch
from dataclasses import dataclass, field

from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import prepare_siamfc_cropping_with_augmentation, apply_siamfc_cropping, \
    apply_siamfc_cropping_to_boxes
from typing import Optional, Sequence, Mapping, Tuple

from trackit.data.utils.collation_helper import collate_element_as_torch_tensor, collate_element_as_np_array
from trackit.data.protocol.train_input import TrainData
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.data import HostDataPipeline
from trackit.core.runtime.metric_logger import get_current_metric_logger

from ..._types import SiameseTrainingPair, SOTFrameInfo, MMOTFrameInfo
from .. import SiameseTrackerTrain_DataTransform
from .augmentation import AugmentationPipeline
from .plugin import ExtraTransform, ExtraTransform_DataCollector


@dataclass(frozen=True)
class SiamFCCroppingParameter:
    output_size: np.ndarray
    area_factor: float
    scale_jitter_factor: float = 0.
    translation_jitter_factor: float = 0.
    output_min_object_size_in_pixel: np.ndarray = field(default_factory=lambda: np.array((0., 0.)))  # (width, height)
    output_min_object_size_in_ratio: float = 0.  # (width, height)
    output_max_object_size_in_pixel: np.ndarray = field(
        default_factory=lambda: np.array((float("inf"), float("inf"))))  # (width, height)
    output_max_object_size_in_ratio: float = 1.  # (width, height)
    interpolation_mode: str = 'bilinear'
    interpolation_align_corners: bool = False


class SiamTrackerTrainingPairProcessor(SiameseTrackerTrain_DataTransform):
    def __init__(self,
                 template_siamfc_cropping_parameter: SiamFCCroppingParameter,
                 search_region_siamfc_cropping_parameter: SiamFCCroppingParameter,
                 augmentation_pipeline: AugmentationPipeline,
                 online_bbox_shift_range_x: Tuple[float, float],
                 online_bbox_shift_range_y: Tuple[float, float],
                 norm_stats_dataset_name: str,
                 additional_processors: Optional[Sequence[ExtraTransform]] = None,
                 visualize: bool = False):
        self.template_siamfc_cropping = SiamFCCropping(template_siamfc_cropping_parameter)
        self.search_region_siamfc_cropping = SiamFCCropping(search_region_siamfc_cropping_parameter)

        self.augmentation_pipeline = augmentation_pipeline
        self.additional_processors = additional_processors

        self.online_bbox_shift_range_x = online_bbox_shift_range_x
        self.online_bbox_shift_range_y = online_bbox_shift_range_y

        self.image_normalize_transform_ = get_dataset_norm_stats_transform(norm_stats_dataset_name, inplace=True)
        self.norm_stats_dataset_name = norm_stats_dataset_name
        self.visualize = visualize

    def __call__(self, training_pair: SiameseTrainingPair, rng_engine: np.random.Generator):
        context = {'is_positive': training_pair.is_positive,
                   'is_online_positive': training_pair.is_online_positive,
                   'z_bbox': training_pair.template.object_bbox,
                   'x_bbox': training_pair.search.object_bbox,
                   'd_bbox': training_pair.online.object_bbox,
                   }

        d_bbox_backup = context['d_bbox'].copy()

        is_online_positive = training_pair.is_online_positive

        # Shift bbox for negative samples
        if not is_online_positive:
            context['d_bbox'] = _shift_bbox_(context['d_bbox'],
                                             self.online_bbox_shift_range_x,
                                             self.online_bbox_shift_range_y,
                                             rng_engine)

        assert self.template_siamfc_cropping.prepare('z', rng_engine, context)
        assert self.template_siamfc_cropping.prepare('d', rng_engine, context)
        if not self.search_region_siamfc_cropping.prepare('x', rng_engine, context):
            return None

        is_positive = training_pair.is_positive

        image_decoding_cache = {}
        if isinstance(training_pair.template, MMOTFrameInfo):
            _decode_mmot_image_with_cache('z', training_pair.template, image_decoding_cache, context)
            _decode_mmot_image_with_cache('x', training_pair.search, image_decoding_cache, context)
            _decode_mmot_image_with_cache('d', training_pair.online, image_decoding_cache, context)
        else:
            _decode_image_with_cache('z', training_pair.template, image_decoding_cache, context)
            _decode_image_with_cache('x', training_pair.search, image_decoding_cache, context)
        del image_decoding_cache

        self.template_siamfc_cropping.do('z', context)
        self.search_region_siamfc_cropping.do('x', context)
        self.template_siamfc_cropping.do('d', context)

        self._do_augmentation(context, rng_engine)

        _bbox_clip_to_image_boundary_(context['z_cropped_bbox'], context['z_cropped_image'])
        _bbox_clip_to_image_boundary_(context['x_cropped_bbox'], context['x_cropped_image'])

        # FIXME: bbox might be invalid here
        try:
            _bbox_clip_to_image_boundary_(context['d_cropped_bbox'], context['d_cropped_image'])
        except:
            print('Invalid shifted bbox found, use original bbox')
            context['d_bbox'] = d_bbox_backup
            assert self.template_siamfc_cropping.prepare('d', rng_engine, context)
            self.template_siamfc_cropping.do('d', context)
            _bbox_clip_to_image_boundary_(context['d_cropped_bbox'], context['d_cropped_image'])

        self.image_normalize_transform_(context['z_cropped_image'])
        self.image_normalize_transform_(context['x_cropped_image'])
        self.image_normalize_transform_(context['d_cropped_image'])

        data = {}

        if self.additional_processors is not None:
            for processor in self.additional_processors:
                processor(training_pair, context, data, rng_engine)

        data['z_cropped_image'] = context['z_cropped_image']
        data['x_cropped_image'] = context['x_cropped_image']
        data['d_cropped_image'] = context['d_cropped_image']

        data['is_positive'] = is_positive
        data['is_online_positive'] = is_online_positive

        if self.visualize:
            from trackit.data.context.worker import get_current_worker_info
            from .visualization import visualize_siam_tracker_training_pair_processor
            output_path = get_current_worker_info().get_output_path()

            if output_path is not None:
                if is_online_positive:
                    output_path = os.path.join(output_path, 'pos')
                else:
                    output_path = os.path.join(output_path, 'neg')
                visualize_siam_tracker_training_pair_processor(output_path, training_pair, context,
                                                               self.norm_stats_dataset_name)

        return data

    def _do_augmentation(self, context: dict, rng_engine: np.random.Generator):
        from .augmentation import AnnotatedImage
        augmentation_context = {'template': [AnnotatedImage(context['z_cropped_image'], context['z_cropped_bbox'])],
                                'search_region': [
                                    AnnotatedImage(context['x_cropped_image'], context['x_cropped_bbox'])],
                                'online': [AnnotatedImage(context['d_cropped_image'], context['d_cropped_bbox'])]}

        self.augmentation_pipeline(augmentation_context, rng_engine)

        context['z_cropped_image'] = augmentation_context['template'][0].image
        context['z_cropped_bbox'] = augmentation_context['template'][0].bbox

        context['x_cropped_image'] = augmentation_context['search_region'][0].image
        context['x_cropped_bbox'] = augmentation_context['search_region'][0].bbox

        context['d_cropped_image'] = augmentation_context['online'][0].image
        context['d_cropped_bbox'] = augmentation_context['online'][0].bbox


def _bbox_clip_to_image_boundary_(bbox: np.ndarray, image: torch.Tensor):
    h, w = image.shape[-2:]
    bbox_clip_to_image_boundary_(bbox, np.array((w, h)))
    assert bbox_is_valid(bbox), f'bbox:\n{bbox}\nimage_size:\n{image.shape}'


def _shift_bbox_(bbox: np.ndarray, shift_range_x: Tuple[float, float], shift_range_y: Tuple[float, float],
                 rng_engine: np.random.Generator):
    width = (bbox[2] - bbox[0])
    height = (bbox[3] - bbox[1])
    dx = rng_engine.uniform(*shift_range_x)
    dy = rng_engine.uniform(*shift_range_y)

    shift_x = dx * width if rng_engine.random() > 0.5 else -dx * width
    shift_y = dy * height if rng_engine.random() > 0.5 else -dy * height

    shift_x = int(shift_x)
    shift_y = int(shift_y)

    new_bbox = bbox.copy()

    new_bbox[0] += shift_x
    new_bbox[1] += shift_y
    new_bbox[2] += shift_x
    new_bbox[3] += shift_y

    new_bbox[0] = max(0, new_bbox[0])
    new_bbox[1] = max(0, new_bbox[1])
    new_bbox[2] = max(new_bbox[0], new_bbox[2])
    new_bbox[3] = max(new_bbox[1], new_bbox[3])

    return new_bbox


class SiamFCCropping:
    def __init__(self, siamfc_cropping_parameter: SiamFCCroppingParameter):
        self.siamfc_cropping_parameter = siamfc_cropping_parameter

    def prepare(self, name: str, rng_engine: np.random.Generator, context: dict):
        cropping_parameter, is_success = \
            prepare_siamfc_cropping_with_augmentation(context[f'{name}_bbox'],
                                                      self.siamfc_cropping_parameter.area_factor,
                                                      self.siamfc_cropping_parameter.output_size,
                                                      self.siamfc_cropping_parameter.scale_jitter_factor,
                                                      self.siamfc_cropping_parameter.translation_jitter_factor,
                                                      rng_engine,
                                                      self.siamfc_cropping_parameter.output_min_object_size_in_pixel,
                                                      self.siamfc_cropping_parameter.output_max_object_size_in_pixel,
                                                      self.siamfc_cropping_parameter.output_min_object_size_in_ratio,
                                                      self.siamfc_cropping_parameter.output_max_object_size_in_ratio)
        if is_success:
            context[f'{name}_cropping_parameter'] = cropping_parameter
        return is_success

    def do(self, name: str, context: dict, normalized: bool = True):
        cropping_parameter = context[f'{name}_cropping_parameter']
        image = context[f'{name}_image']
        image_cropped, context[f'{name}_image_mean'], cropping_parameter = \
            apply_siamfc_cropping(image, self.siamfc_cropping_parameter.output_size, cropping_parameter,
                                  interpolation_mode=self.siamfc_cropping_parameter.interpolation_mode,
                                  align_corners=self.siamfc_cropping_parameter.interpolation_align_corners)
        if normalized:
            image_cropped.div_(255.)
        context[f'{name}_cropping_parameter'] = cropping_parameter
        context[f'{name}_cropped_image'] = image_cropped
        context[f'{name}_cropped_bbox'] = apply_siamfc_cropping_to_boxes(context[f'{name}_bbox'], cropping_parameter)


def _decode_image_with_cache(name: str, frame: SOTFrameInfo, cache: dict, context: dict):
    if frame.image in cache:
        context[f'{name}_image'] = cache[frame.image]
        return
    image = frame.image()

    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1)).contiguous()
    image = image.to(torch.float32)

    cache[frame.image] = image
    context[f'{name}_image'] = image


def _decode_mmot_image_with_cache(name: str, frame: MMOTFrameInfo, cache: dict, context: dict):
    try:
        if frame.image[0].args[1] in cache:
            context[f'{name}_image'] = cache[frame.image]
            return
        image_v = frame.image[0]()
        image_i = frame.image[1]()
        image = np.concatenate([image_v, image_i], axis=-1)

        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1)).contiguous()
        image = image.to(torch.float32)

        cache[frame.image[0]] = image
        context[f'{name}_image'] = image

    except:
        # NOTE: For non parallel mode
        image_v = frame.image[0]
        image_i = frame.image[1]
        image = np.concatenate([image_v, image_i], axis=-1)

        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1)).contiguous()
        image = image.to(torch.float32)

        context[f'{name}_image'] = image


class SiamTrackerTrainingPairProcessorBatchCollator:
    def __init__(self, additional_collators: Optional[Sequence[ExtraTransform_DataCollector]] = None):
        self.additional_collators = additional_collators

    def __call__(self, batch: Sequence[Mapping], collated: TrainData):
        collated.input.update({
            'z': collate_element_as_torch_tensor(batch, 'z_cropped_image'),
            'x': collate_element_as_torch_tensor(batch, 'x_cropped_image'),
            'd': collate_element_as_torch_tensor(batch, 'd_cropped_image')
        })
        collated.miscellanies.update({'is_positive': collate_element_as_np_array(batch, 'is_positive')})

        if self.additional_collators is not None:
            for additional_collator in self.additional_collators:
                additional_collator(batch, collated)


class SiamTrackerTrainingPairProcessorHostLoggingHook(HostDataPipeline):
    def pre_process(self, input_data: TrainData) -> TrainData:
        is_positive = input_data.miscellanies['is_positive']
        positive_sample_ratio = (np.sum(is_positive) / len(is_positive)).item()
        get_current_metric_logger().log({'positive_pair': positive_sample_ratio})
        return input_data
