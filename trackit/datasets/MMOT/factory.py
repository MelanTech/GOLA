from typing import List, Iterable, Optional
from trackit.datasets.common.factory import DatasetFactory
from trackit.datasets.base.video.dataset import VideoDataset
from trackit.datasets.base.video.filter.func import apply_filters_on_video_dataset_
from .constructor import MultiModalObjectTrackingDatasetConstructorGenerator
from .specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
from .specialization.memory_mapped.constructor import \
    construct_multi_modal_object_tracking_dataset_memory_mapped_from_base_video_dataset

__all__ = ['MultiModalObjectTrackingDatasetFactory']


class MultiModalObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: Iterable):
        super(MultiModalObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                     MultiModalObjectTrackingDatasetConstructorGenerator,
                                                                     apply_filters_on_video_dataset_,
                                                                     MultiModalObjectTrackingDataset_MemoryMapped,
                                                                     construct_multi_modal_object_tracking_dataset_memory_mapped_from_base_video_dataset)

    def construct(self, filters: Optional[Iterable] = None, cache_base_format: bool = True,
                  dump_human_readable: bool = False) -> List[MultiModalObjectTrackingDataset_MemoryMapped]:
        return super(MultiModalObjectTrackingDatasetFactory, self).construct(filters, cache_base_format,
                                                                             dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(MultiModalObjectTrackingDatasetFactory, self).construct_base_interface(filters, make_cache,
                                                                                            dump_human_readable)
