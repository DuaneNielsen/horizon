from markup_video import Line
import numpy as np
from torchvision.io import read_image
from torchvision.transforms.functional import rotate, resize, normalize
from math import ceil, degrees
import torch
import json


class HorizonDataSet:
    def __init__(self, data_dir='data/horizon', rotate=True, num_classes=16, select_label=None, image_size=32):
        self.data_dir = data_dir
        self.rotate = rotate
        self.bins = num_classes
        self.select_label = select_label
        self.image_size = image_size

        with open(f'{self.data_dir}/lines.json') as f:
            lines = json.loads(f.read())
        self.lines = lines
        self.index = list(lines)

        # mask
        x = torch.linspace(-self.image_size // 2, self.image_size // 2, self.image_size)
        x, y = torch.meshgrid([x, x], indexing='ij')
        self.mask = torch.sqrt(x ** 2 + y ** 2).lt(self.image_size // 2)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):

        img = read_image(f'{self.data_dir}/{self.index[item]}.png').float()

        # resize
        h = self.image_size
        orig_h, orig_w = img.shape[1], img.shape[2]
        rescale_x_factor = orig_h / orig_w
        scale = orig_h / h
        img = resize(img, [h, h])
        img = normalize(img, [img[i].mean() for i in range(3)], [img[i].std() for i in range(3)], inplace=True)

        # load the serialized form into a numpy array
        lines = np.stack([Line.from_flat(line).to_numpy() for line in self.lines[self.index[item]]], axis=-1)
        lines[0, :, :] = lines[0, :, :] * rescale_x_factor
        lines[:, :, :] = lines[:, :, :] / scale

        # compute l2 norm for each line ( rise - run in normal form ), then convert to complex plane, and angle
        slope = (lines[:, 1, :] - lines[:, 0, :])
        length = np.linalg.norm(slope, ord=2, axis=0, keepdims=True)
        complex = (slope / length).T.copy().view(np.complex128)
        complex_mean = complex.mean()
        angle = np.angle(complex_mean)

        # data augmentation - rotate
        if self.rotate:
            d_angle = torch.rand(1) * 2 * torch.pi
            img = rotate(img, angle=degrees(d_angle.item()), center=[img.shape[2] / 2, img.shape[1] / 2])
            angle = angle - d_angle
            angle = (angle + 2 * np.pi) % (2 * np.pi)

        complex_mean = torch.complex(real=torch.cos(angle), imag=torch.sin(angle))

        img = img * self.mask.unsqueeze(0)

        # calculate class
        slise_size = (2 * np.pi) / self.bins
        discrete = int(np.floor(angle / slise_size))

        labels = {
            'lines': lines,
            'complex': complex,
            'complex_mean': complex_mean,
            'angle': angle,
            'discrete': discrete,
            'discrete_min': discrete * slise_size,
            'discrete_max': (discrete + 1) * slise_size
        }

        if self.select_label is not None:
            return img, labels[self.select_label]
        else:
            return img, labels
