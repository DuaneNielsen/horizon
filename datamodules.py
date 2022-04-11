from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, dataloader
from markup_video import Line
import numpy as np
from torchvision.io import read_image
from torchvision.transforms.functional import rotate, resize, crop
from math import ceil, degrees
import torch
import json


class HorizonDataSet:
    def __init__(self, data_dir='data/horizon', rotate=True, bins=360, select_label=None):
        self.data_dir = data_dir
        with open(f'{self.data_dir}/lines.json') as f:
            lines = json.loads(f.read())
        self.lines = lines
        self.index = list(lines)
        self.rotate = rotate
        self.bins = bins
        self.select_label = select_label

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):

        img = read_image(f'{self.data_dir}/{self.index[item]}.png')

        # resize
        h = 64
        orig_h, orig_w = img.shape[1], img.shape[2]
        rescale_x_factor = orig_h/orig_w
        scale = orig_h / h
        img = resize(img, [h, h])

        # mask
        x = torch.linspace(-h//2, h//2, h)
        x, y = torch.meshgrid([x, x])
        d = torch.sqrt(x ** 2 + y ** 2)
        mask = d < h//2

        # load the serialized form into a numpy array
        lines = np.stack([Line.from_flat(line).to_numpy() for line in self.lines[self.index[item]]], axis=-1)
        lines[0, :, :] = lines[0, :, :] * rescale_x_factor
        lines[:, :, :] = lines[:, :, :] / scale

        # compute l2 norm for each line ( rise - run in normal form ), then convert to complex plane, and angle
        slope = (lines[:, 1, :] - lines[:, 0, :])
        length = np.linalg.norm(slope, ord=2, axis=0, keepdims=True)
        complex = (slope / length).T.copy().view(np.complex128)
        angle = np.angle(complex.mean())

        # data augmentation - rotate
        if self.rotate:
            d_angle = torch.rand(1) * 2 * np.pi
            img = rotate(img, angle=degrees(d_angle.item()), center=[img.shape[2]/2, img.shape[1]/2])
            angle = angle - d_angle
            angle = (angle + 2 * np.pi) % (2 * np.pi)

        img = img * mask.unsqueeze(0)

        # calculate class
        slise_size = self.bins / (2 * np.pi)
        discrete = int(np.floor(angle * slise_size))

        labels = {
            'lines': lines,
            'complex': complex,
            'angle': angle,
            'discrete': discrete
        }

        if self.select_label is not None:
            return img.float(), labels[self.select_label]
        else:
            return img.float(), labels


class HorizonAnglesDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_classes=90, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.train = None
        self.val = None
        self.setup()

    def setup(self, stage=None):
        full = HorizonDataSet(self.data_dir, select_label='discrete', bins=90)
        train_size, val_size = len(full) * 9 // 10, ceil(len(full) / 10)
        self.train, self.val = random_split(full, [train_size, val_size])

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return dataloader.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

