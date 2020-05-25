import collections
import os
import os.path

import cv2
import numpy as np
import torch
from glob import glob
from torch.utils import data
from PIL import Image

KITTI_CLASSES = [
    'BG', 'Car', 'Van', 'Truck',
    'Pedestrian', 'Person_sitting',
    'Cyclist', 'Tram', 'Misc', 'DontCare'
]
KITTI_CLASSES_PRIORITY = [
   "Car", 'Pedestrian', 'Truck',
   'Van', 'Tram', 'Cyclist'
]
NUM_TRAIN_SEQUENCES = 15


class Class_to_ind(object):
    def __init__(self, binary, binary_item):
        self.binary = binary
        self.binary_item = binary_item
        self.classes = KITTI_CLASSES

    def __call__(self, name):
        if not name in self.classes:
            raise ValueError('No such class name : {}'.format(name))
        else:
            if self.binary:
                if name == self.binary_item:
                    return True
                else:
                    return False
            else:
                return self.classes.index(name)


# def get_data_path(name):
#     js = open('config.json').read()
#     data = json.loads(js)
#     return data[name]['data_path']

class AnnotationTransform_kitti(object):
    '''
    Transform Kitti detection labeling type to norm type:
    source: Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62
    target: [xmin,ymin,xmax,ymax,label_ind]
    levels=['easy','medium']
    '''

    def __init__(self,  observed_classes, class_to_ind=Class_to_ind(False, None),
                 levels=['easy', 'medium', 'hard']):
        self.class_to_ind = class_to_ind
        self.observed_classes = observed_classes
        self.levels = levels if isinstance(levels, list) else [levels]

    def __call__(self, target_lines, width, height):

        res = list()
        boxes = []
        labels = []
        for line in target_lines:
            line_fields = line.strip().split(' ')
            occlusion = int(line_fields[4])
            if occlusion != 0:
                continue

            xmin, ymin, xmax, ymax = tuple(line_fields[6:10])

            bnd_box = [xmin, ymin, xmax, ymax]
            new_bnd_box = bnd_box
            for i, pt in enumerate(range(4)):
                cur_pt = float(bnd_box[i])
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                new_bnd_box.append(cur_pt)
            label = line_fields[2]
            if label not in self.observed_classes:
                continue
            label_idx = self.class_to_ind(label)
            boxes.append(bnd_box)
            labels.append(label_idx)
        return torch.Tensor(boxes), torch.Tensor(labels)


class KittiLoader(data.Dataset):
    def __init__(self, root, image_set="train",
                 transforms=None, num_classes=4):
        self.root = root
        self.image_set = image_set
        self.n_classes = min(num_classes, len(KITTI_CLASSES_PRIORITY))
        self.observed_classes = set(KITTI_CLASSES_PRIORITY[:self.n_classes])
        self.target_transform = AnnotationTransform_kitti(self.observed_classes)
        #self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.transforms = transforms
        self.name = 'kitti'

        image_root = os.path.join(root, "training", "image_2")
        sequences = os.listdir(image_root)
        if self.image_set == 'train':
            sequences = sequences[:NUM_TRAIN_SEQUENCES]
        else:
            sequences = sequences[NUM_TRAIN_SEQUENCES:]
        self.files = []
        self.labels = []
        for sequence in sequences:
            self.files.extend(glob(os.path.join(image_root, sequence, '*.png')))
            self.labels.extend(glob(os.path.join(root, "training", 'label_2', '*.txt')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = img_name

        # img = m.imread(img_path)
        img = Image.open(img_path).convert("RGB")
        height, width, channels = img.shape
        # img = np.array(img, dtype=np.uint8)

        lbl_path = self.labels[index]
        lbl_lines = open(lbl_path, 'r').readlines()
        boxes, labels = self.target_transform(lbl_lines, width, height)

        # if self.is_transform:
        #     img, lbl = self.transform(img, lbl)

        target = {}
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # target entry must be a dict containing array of ground truth boxes under key 'boxes'
        # https://github.com/pytorch/vision/blob/2875315d6b6b4f5a375f04b6673ef2a57483edfa/torchvision/models/detection/rpn.py#L462
        target["boxes"] = boxes
        target["labels"] = labels
        # labels are read at
        # https://github.com/pytorch/vision/blob/2875315d6b6b4f5a375f04b6673ef2a57483edfa/torchvision/models/detection/roi_heads.py#L734
        target["image_id"] = index
        target["area"] = area

        # Empty field for interface with coco
        target["iscrowd"] = torch.tensor(0)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

        if self.transforms is not None:
            target = np.array(target)
            img, boxes, labels = self.transforms(img, target[:, :4], target[:, 4])
            # img, lbl = self.transforms(img, lbl)
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.image_set != "testing":
            # return img, lbl
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        else:
            return img
