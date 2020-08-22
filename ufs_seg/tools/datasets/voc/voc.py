from datasets.BaseDataset import BaseDataset


class VOC(BaseDataset):
    @classmethod
    def get_class_colors(*args):
        return [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128],
                [0, 0, 64], [0, 0, 192], [0, 128, 64],
                [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0],
                [0, 192, 128], [128, 64, 0], ]

    @classmethod
    def get_class_names(*args):
        return ['background', 'aeroplane', 'bicycle', 'bird',
                'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable',
                'dog', 'horse', 'motorbike', 'person',
                'pottedplant',
                'sheep', 'sofa', 'train', 'tv/monitor']
