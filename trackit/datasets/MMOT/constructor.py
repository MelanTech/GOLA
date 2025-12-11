from trackit.datasets.base.common.utils.bounding_box import set_bounding_box_
from trackit.datasets.base.common.constructor import BaseDatasetConstructorGenerator, set_mmot_path_, \
    BaseDatasetSequenceConstructorGenerator, BaseDatasetSequenceConstructor, BaseVideoDatasetConstructor, \
    convert_sequence_paths_as_posix_stype, generate_mmot_sequence_path_

__all__ = ['MultiModalObjectTrackingDatasetConstructor', 'MultiModalObjectTrackingDatasetConstructorGenerator']

from trackit.miscellanies.operating_system.interface import IS_WIN32


class MultiModalObjectTrackingDatasetFrameConstructor:
    def __init__(self, frame, root_path, context):
        self.frame = frame
        self.root_path = root_path
        self.object = None
        self.context = context

    def _try_allocate_object(self):
        if self.object is None:
            self.object = {'id': 0}
            self.frame['objects'] = [self.object]

    def set_path(self, path: str, image_size=None):
        assert path is not None and isinstance(path, tuple), f'invalid path {path}'
        set_mmot_path_(self.frame, path, self.root_path, image_size)

    def set_bounding_box(self, bounding_box, validity=None, dtype=None):
        self._try_allocate_object()
        set_bounding_box_(self.object, bounding_box, validity, dtype, self.context)

    def set_object_attribute(self, name, value):
        assert isinstance(name, str), f'invalid name {name}'
        self._try_allocate_object()
        self.object[name] = value

    def set_frame_attribute(self, name, value):
        assert isinstance(name, str), f'invalid name {name}'
        self.frame[name] = value


class MultiModalObjectTrackingDatasetFrameConstructorGenerator:
    def __init__(self, frame, root_path, context):
        self.frame = frame
        self.root_path = root_path
        self.context = context

    def __enter__(self):
        return MultiModalObjectTrackingDatasetFrameConstructor(self.frame, self.root_path, self.context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultiModalObjectTrackingDatasetSequenceConstructor(BaseDatasetSequenceConstructor):
    def __init__(self, sequence: dict, root_path: str, context):
        super(MultiModalObjectTrackingDatasetSequenceConstructor, self).__init__(sequence, root_path, context)
        self.object_ = None
        if 'objects' in self.sequence:
            self.object_ = self.sequence['objects'][0]

    def new_frame(self):
        frame = {}
        if 'frames' not in self.sequence:
            self.sequence['frames'] = []
        self.sequence['frames'].append(frame)
        return MultiModalObjectTrackingDatasetFrameConstructorGenerator(frame, self.root_path, self.context)

    def open_frame(self, index: int):
        return MultiModalObjectTrackingDatasetFrameConstructorGenerator(self.sequence['frames'][index], self.root_path,
                                                                        self.context)

    def set_object_attribute(self, name: str, value):
        if self.object_ is None:
            self.object_ = {name: value, 'id': 0}
            self.sequence['objects'] = [self.object_]
        else:
            self.object_[name] = value


class MultiModalObjectTrackingDatasetSequenceConstructorGenerator(BaseDatasetSequenceConstructorGenerator):
    def __init__(self, sequence, root_path, category_id, context):
        super(MultiModalObjectTrackingDatasetSequenceConstructorGenerator, self).__init__(sequence, context)

        self.root_path = root_path
        sequence_object_attributes = [{'id': 0}]
        if category_id is not None:
            sequence_object_attributes[0]['category_id'] = category_id

        self.sequence['objects'] = sequence_object_attributes

    def __enter__(self):
        return MultiModalObjectTrackingDatasetSequenceConstructor(self.sequence, self.root_path, self.context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert 'path' not in self.sequence
        generate_mmot_sequence_path_(self.sequence)
        if IS_WIN32:
            convert_sequence_paths_as_posix_stype(self.sequence)
        self.context.get_processing_bar().update()


class MultiModalObjectTrackingDatasetConstructor(BaseVideoDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, version: int, context):
        super(MultiModalObjectTrackingDatasetConstructor, self).__init__(dataset, root_path, version, context)
        if 'sequences' not in dataset:
            dataset['sequences'] = []

    def new_sequence(self, category_id=None):
        if category_id is not None:
            assert category_id in self.dataset['category_id_name_map']
        sequence = {}
        self.dataset['sequences'].append(sequence)
        return MultiModalObjectTrackingDatasetSequenceConstructorGenerator(sequence, self.root_path, category_id,
                                                                           self.context)


class MultiModalObjectTrackingDatasetConstructorGenerator(BaseDatasetConstructorGenerator):
    def __init__(self, dataset: dict, root_path: str, version: int):
        super(MultiModalObjectTrackingDatasetConstructorGenerator, self).__init__(dataset, root_path, version,
                                                                                  MultiModalObjectTrackingDatasetConstructor)
