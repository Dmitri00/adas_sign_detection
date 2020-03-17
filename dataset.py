import torch
from torchvision import datasets, transforms, models
import os
from PIL import Image
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
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.gt_name = os.path.join(root, "gt.txt")
        files_in_dir = os.listdir(root)
        self.imgs = list(sorted(filter(lambda x: 'ppm' in x, files_in_dir)))
        with open(self.gt_name, 'r') as gt_file:
            gt_lines = gt_file.readlines()
        self.ground_truth = self.parse_ground_truth(gt_lines)
    def parse_ground_truth(self, gt_lines):
        gt_dict = {}
        class_counter = {}
        for entry_line in gt_lines:
            entry_splited = entry_line.split(';')
            img_name = entry_splited[0]
            box = list(map(int, entry_splited[1:5]))
            box_class = int(entry_splited[5])
            if box_class not in class_counter:
                class_counter[box_class] = 1
            
            if img_name in gt_dict:
                gt_dict[img_name].append((box, box_class))
            else:
                gt_dict[img_name] = [(box, box_class)]
        self.num_classes = len(class_counter.keys())
        return gt_dict

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])
        box = 0
        label = 1
        boxes = [self.ground_truth[idx][sign][box] for sign in self.ground_truth[idx]]
        labels = [self.ground_truth[idx][sign][label] for sign in self.ground_truth[idx]]
        img = Image.open(img_path).convert("RGB")
        

        # convert everything into a torch.Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        # target entry must be a dict containing array of ground truth boxes under key 'boxes'
        # https://github.com/pytorch/vision/blob/2875315d6b6b4f5a375f04b6673ef2a57483edfa/torchvision/models/detection/rpn.py#L462
        target["boxes"] = boxes
        target["labels"] = labels
        # labels are read at
        # https://github.com/pytorch/vision/blob/2875315d6b6b4f5a375f04b6673ef2a57483edfa/torchvision/models/detection/roi_heads.py#L734
        target["image_id"] = image_id
        target["area"] = area


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)