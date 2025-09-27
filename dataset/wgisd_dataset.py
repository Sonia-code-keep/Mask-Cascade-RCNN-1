import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from glob import glob

from structures import Boxes, Instances, BitMasks
opj = os.path.join

cls2idx = {
    'CDY': 0,
    'CFR': 1,
    'CSV': 2,
    'SVB': 3,
    'SYH': 4,
}

idx2cls = {v: k for k, v in cls2idx.items()}


def train_aug(SIZE):
    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.RandomResizedCrop(height=SIZE, width=SIZE, always_apply=True),  # can't handle mask
        # A.Cutout(num_holes=8, max_h_size=94, max_w_size=94, fill_value=0, p=0.5),  # up
        A.Resize(height=SIZE[0], width=SIZE[1], always_apply=True),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return aug


def val_aug(SIZE):
    aug = A.Compose([
        A.Resize(height=SIZE[0], width=SIZE[1], p=1.),
        A.Normalize(),
        ToTensorV2()
    ])
    return aug

from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np

class CocoFloorPlanDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, path)).convert("RGB")

        boxes, masks, labels = [], [], []
        for ann in anns:
            boxes.append(ann['bbox'])  # [x, y, w, h]
            masks.append(self.coco.annToMask(ann))
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        return img, boxes, labels, masks

    def __len__(self):
        return len(self.ids)







class WSGISDDataset(Dataset):
    def __init__(self, root, mode='train', resize=(1024, 1024)):
        self.root = root
        self.data_root = opj(self.root, 'data')
        with open(opj(root, f'{mode}_masked.txt')) as f:
            self.samples = f.read().split('\n')

        self.samples = np.array(self.samples)
        keep = np.ones(len(self.samples))
        for i in range(len(self.samples)):
            if len(self.samples[i]) <= 1:
                keep[i] = 0
        self.samples = self.samples[keep == 1]

        self.mode = mode
        self.resize = resize  # h, w
        self.aug = train_aug(self.resize) if mode == 'train' else val_aug(self.resize)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(opj(self.data_root, f'{sample}.jpg'))
        meta = {}
        meta['file_name'] = sample
        meta['resize'] = self.resize
        meta['width'] = int(img.width)
        meta['height'] = int(img.height)
        img = np.array(img)
        if self.mode in ['train', 'test']:  # return gt
            with open(opj(self.data_root, f'{sample}.txt')) as f:
                box_data = f.read().split('\n')[:-1]
            boxes = np.zeros((len(box_data), 4))
            for i, b in enumerate(box_data):
                _, cx, cy, w, h = [float(i) for i in b.split(' ')]
                half_h, half_w = h / 2, w / 2
                x1 = max(cx - half_w, 0)
                y1 = max(cy - half_h, 0)
                x2 = min(cx + half_w, 1)
                y2 = min(cy + half_h, 1)
                boxes[i] = np.array([x1, y1, x2, y2])
            boxes[:, [0, 2]] *= meta['width']
            boxes[:, [1, 3]] *= meta['height']

            clses = np.array([cls2idx[sample[:3]]] * len(boxes))
            masks = np.load(opj(self.data_root, f'{sample}.npz'))['arr_0']  # [H, W, N]

            sample = self.aug(**{'image': img, 'bboxes': boxes, 'labels': clses, 'mask': masks})
            _img, _boxes, _clses, _masks = \
                sample['image'], torch.tensor(sample['bboxes']), torch.tensor(sample['labels']), sample['mask']

            keep = torch.zeros(_masks.shape[-1]).bool()
            for i in range(_masks.shape[-1]):
                keep[i] = _masks[..., i].sum().item() > 10
            _masks = _masks.permute(2, 0, 1)[keep]
            _boxes = _boxes[keep]
            _clses = _clses[keep]

            assert len(_boxes) == len(_clses) == _masks.size(0), \
                f'file:{self.samples[idx]} box:{len(_boxes)} cls:{len(_clses)} mask:{_masks.size(0)}'
            return _img, _boxes, _clses, _masks, meta
        else:  # infernce, no gt
            pass

    def collate_fn(self, batch):
        img_list, box_list, cls_list, mask_list, meta_list = zip(*batch)
        assert len(img_list) == len(cls_list) == len(box_list) == len(mask_list) == len(meta_list)
        batched_inputs = []
        for i in range(len(img_list)):
            _dict = {}
            _dict['file_name'] = meta_list[i]['file_name']
            _dict['height'] = meta_list[i]['height']
            _dict['width'] = meta_list[i]['width']

            _dict['image'] = img_list[i]
            _dict['instances'] = Instances(image_size=(meta_list[i]['resize'][0], meta_list[i]['resize'][0]))
            _dict['instances'].gt_boxes = Boxes(box_list[i])
            _dict['instances'].gt_classes = cls_list[i].long()
            _dict['instances'].gt_masks = BitMasks(mask_list[i])

            batched_inputs.append(_dict)
        return batched_inputs


