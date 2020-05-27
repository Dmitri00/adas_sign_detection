import collections
import os
import os.path
from bisect import bisect_right

import numpy as np
import pandas as pd
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

        boxes = []
        labels = []
        prev_frame_id = 0
        targets =

        for line in target_lines:
            line_fields = line.strip().split(' ')

            frame_id = int(line_fields[0])

            if frame_id != prev_frame_id:
                frame = {"boxes": torch.Tensor(boxes), "labels": torch.Tensor(labels),
                         "frame_id": prev_frame_id}
                targets.append(frame)
                boxes = []
                labels = []
            prev_frame_id = frame_id

            occlusion = int(line_fields[4])

            label = line_fields[2]
            label_idx = self.class_to_ind(label)

            xmin, ymin, xmax, ymax = tuple(line_fields[6:10])
            bnd_box = [xmin, ymin, xmax, ymax]

            if occlusion != 0:
                continue
            if label not in self.observed_classes:
                continue

            labels.append(label_idx)
            boxes.append(bnd_box)

        frame = {"boxes": torch.Tensor(boxes),
                 "labels": torch.Tensor(labels),
                 "frame_id": frame_id}
        targets.append(frame)

        return targets
class AnnotationImporter(AnnotationTransform_kitti):
    def __call__(self, annotation_file, img_files_list):
        files_df = pd.Series(img_files_list, name="file")
        files_df.dropna()
        names = 'frame track_id type truncated occluded alpha x0 y0 x1 y1 height width length loc_x loc_y loc_z phi'.split(
            ' ')
        annot_df = pd.read_csv(annotation_file,
                               header=None, sep=' ',
                               names=names)

        keep = annot_df["occluded"] == 1
        annot_df = annot_df[keep]

        keep = annot_df["type"].isin(self.observed_classes)
        annot_df = annot_df[keep]

        for col in ("x0", "y0", "x1", "y1"):
            keep = 1 < annot_df[col]
            annot_df = annot_df[keep]
            # annot_df[col] = annot_df[annot_df[col] < 5]*5
            keep = annot_df[col] < 1400
            annot_df = annot_df[keep]

        annot_df["width"] = (annot_df["x1"] - annot_df["x0"])
        keep = annot_df["width"] > 10
        annot_df = annot_df[keep]
        annot_df["height"] = (annot_df["y1"] - annot_df["y0"])
        keep = annot_df["height"] > 10
        annot_df = annot_df[keep]
        type_categories = pd.Categorical(annot_df["type"], categories=KITTI_CLASSES)
        annot_df["type"] = type_categories.codes
        annot_df["box"] = list(map(np.array, zip(annot_df["x0"], annot_df["y0"], annot_df["x1"], annot_df["y1"])))

        # full_annotation = annot_df.join(files_df, how="left")

        grouped_df = annot_df.groupby(["frame"])

        boxes = grouped_df["box"].apply(np.array)
        labels = grouped_df["type"].apply(np.array)
        # files = grouped_df["file"].apply(n)
        # import pdb; pdb.set_trace()
        target_df = pd.concat((boxes, labels), axis=1)
        target_df = target_df.join(files_df, on="frame", how="inner")
        # train_records = grouped_df[["box", "type", "file"]].apply(lambda x: list(pd.DataFrame.to_records(x)))
        return target_df

class KittiLoader(data.Dataset):
    def __init__(self, root, image_set="train",
                 transforms=None, num_classes=4):
        self.root = root
        self.image_set = image_set
        self.n_classes = min(num_classes, len(KITTI_CLASSES_PRIORITY))
        self.observed_classes = set(KITTI_CLASSES_PRIORITY[:self.n_classes])
        self.target_transform = AnnotationImporter(self.observed_classes)
        #self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.transforms = transforms
        self.name = 'kitti'

        image_root = os.path.join(root, "training", "image_02")
        #sequences = os.listdir(image_root)
        sequences = glob(os.path.join(root, "training", 'label_02', '*.txt'))
        sequences = sorted(sequences)
        if self.image_set == 'train':
            sequences = sequences[:NUM_TRAIN_SEQUENCES]
        else:
            sequences = sequences[NUM_TRAIN_SEQUENCES:]
        self.files = []
        self.sequence_lengths = []
        self.labels = []
        cumulative_length = 0
        for sequence in sequences:
            sequence_name = os.path.basename(sequence)

            files_in_seq = glob(os.path.join(image_root, sequence_name, '*.png'))
            self.sequence_lengths.append(cumulative_length)
            with open(sequence, 'r') as label_file:
                sequence_targets = self.target_transform(label_file, files_in_seq)

            #self.files.extend(sorted(files_in_seq))
            self.labels = pd.concat((self.labels, sequence_targets))
            cumulative_length += len(files_in_seq)

    def get_sequence_of_frame(self, index):
        return bisect_right(self.sequence_lengths, index) - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        targets_df = self.labels.iloc[index]
        boxes = torch.Tensor(targets_df["box"])
        labels = torch.LongTensor(targets_df["type"])
        img_path = targets_df["file"]
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            return self.__getitem__(index+1)


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
