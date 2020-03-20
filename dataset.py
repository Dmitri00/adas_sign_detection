import torch
from torchvision import datasets, transforms, models
import os
from PIL import Image
import numpy as np
data_transforms = {
    'train': [transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ],
    'test': [transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ]
}




class GermanTrafficDataset(object):
    
    def __init__(self, root, transforms, image_set):
        if image_set == 'train':
            self.root = os.path.join(root, 'TrainIJCNN2013')
        elif image_set == 'val' or image_set == 'test':
            self.root = os.path.join(root, 'TestIJCNN2013') 
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.gt_name = os.path.join(self.root, "gt.txt")
        files_in_dir = os.listdir(self.root)
        self.imgs = list(sorted(filter(lambda x: 'ppm' in x, files_in_dir)))
        with open(self.gt_name, 'r') as gt_file:
            gt_lines = gt_file.readlines()
        self.ground_truth = self.parse_ground_truth(gt_lines)
    def map_box_class(self, box_class):
        return box_class
    def parse_ground_truth(self, gt_lines):
        gt_dict = {}
        class_counter = {}
        for entry_line in gt_lines:
            entry_splited = entry_line.split(';')
            img_name = entry_splited[0]
            box = list(map(int, entry_splited[1:5]))
            box_class = self.map_box_class(int(entry_splited[5]))
            if box_class not in class_counter:
                class_counter[box_class] = 1
            
            if img_name in gt_dict:
                gt_dict[img_name].append((box, box_class))
            else:
                gt_dict[img_name] = [(box, box_class)]
        self.num_classes = len(class_counter.keys())
        
        self.imgs = list(gt_dict.keys())
        ##()
        return gt_dict

    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        box = 0
        label = 1
        boxes = [sign[box] for sign in self.ground_truth[img_name]]
        labels = [sign[label] for sign in self.ground_truth[img_name]]
        img = Image.open(img_path).convert("RGB")
        

        # convert everything into a torch.Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        # target entry must be a dict containing array of ground truth boxes under key 'boxes'
        # https://github.com/pytorch/vision/blob/2875315d6b6b4f5a375f04b6673ef2a57483edfa/torchvision/models/detection/rpn.py#L462
        target["boxes"] = boxes
        target["labels"] = labels
        # labels are read at
        # https://github.com/pytorch/vision/blob/2875315d6b6b4f5a375f04b6673ef2a57483edfa/torchvision/models/detection/roi_heads.py#L734
        target["image_id"] = image_id
        target["area"] = area

        # Empty field for interface with coco
        target["iscrowd"] = torch.tensor(0)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class GermanTrafficDataset4(GermanTrafficDataset):
    meta_classes = { 'prohibitory': set([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]),
                        'mandatory': set([33, 34, 35, 36, 37, 38, 39, 40]),
                        'danger': set([11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
                        'other': set([6, 12, 13, 14, 17, 32, 41, 42])
        }
    def map_box_class(self, box_class):
        for meta_class_i, (meta_class, values) in enumerate(self.meta_classes.items()):
            if box_class in values:
                return meta_class_i

class PennFudanDataset(object):
    def __init__(self, root, transforms, image_set):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        indices = torch.randperm(len(self.imgs)).tolist()
        if image_set == 'train':
            self.imgs = [self.imgs[ind] for ind in indices[:50]]
            self.masks = [self.masks[ind] for ind in indices[:50]]
        else:
            self.imgs = [self.imgs[ind] for ind in indices[-50:]]
            self.masks = [self.masks[ind] for ind in indices[-50:]]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)