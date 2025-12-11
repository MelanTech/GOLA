# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add LasHeR dataset

import os
import numpy as np

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.MMOT.constructor import MultiModalObjectTrackingDatasetConstructor


class LasHeR_Seed(BaseSeed):
    def __init__(self, root_path: str = None, data_split: str = ('train', 'test')):
        if root_path is None:
            # get the path from `consts.yaml` file
            root_path = self.get_path_from_config('LasHeR_PATH')

        super(LasHeR_Seed, self).__init__(
            'LasHeR',  # dataset name
            root_path,  # dataset root path
            supported_data_splits=('train', 'test'),
            data_split=data_split,
        )

    def construct(self, constructor: MultiModalObjectTrackingDatasetConstructor):
        # Implement the dataset construction logic here
        if self.data_split[0] == 'train':
            root_path = os.path.join(self.root_path, 'trainingset')
            with open('{}trainingsetList.txt'.format(self.root_path)) as f:
                sequence_names = f.read().splitlines()
        elif self.data_split[0] == 'test':
            root_path = os.path.join(self.root_path, 'testingset')
            with open('{}testingsetList.txt'.format(self.root_path)) as f:
                sequence_names = f.read().splitlines()
        else:
            raise NotImplementedError('Incorrect data split')

        # Set the total number of sequences (Optional, for progress bar)
        constructor.set_total_number_of_sequences(len(sequence_names))

        # Set the bounding box format (Optional, 'XYXY' or 'XYWH', default for XYWH)
        constructor.set_bounding_box_format('XYWH')

        for sequence_name in sequence_names:
            with constructor.new_sequence() as sequence_constructor:
                sequence_constructor.set_name(sequence_name)

                sequence_path = os.path.join(root_path, sequence_name)
                # groundtruth.txt: the path of the bounding boxes file
                boxes_path = os.path.join(sequence_path, 'init.txt')
                frames_path_v = os.path.join(sequence_path, 'visible')
                frames_path_i = os.path.join(sequence_path, 'infrared')

                # load bounding boxes using numpy
                boxes = np.loadtxt(boxes_path, delimiter=',')

                frame_ids_v = sorted(os.listdir(frames_path_v))
                frame_ids_i = sorted(os.listdir(frames_path_i))

                for frame_id_v, frame_id_i, box in zip(frame_ids_v, frame_ids_i, boxes):
                    # frame_path: the path of the frame image,
                    # assuming the frame image is named as 0001.jpg, 0002.jpg, ...

                    frame_path_v = os.path.join(frames_path_v, frame_id_v)
                    frame_path_i = os.path.join(frames_path_i, frame_id_i)

                    with sequence_constructor.new_frame() as frame_constructor:
                        # set the frame path and image size
                        # image_size is optional (will be read from the image if not provided)
                        frame_constructor.set_path((frame_path_v, frame_path_i))
                        # set the bounding box
                        # validity is optional (False for fully occluded or out-of-view or not annotated)
                        frame_constructor.set_bounding_box(box, validity=True)
