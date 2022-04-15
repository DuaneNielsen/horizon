from torch.utils.data.dataset import T_co

from markup_video import Line
import numpy as np
from torchvision.io import read_image
from torchvision.transforms.functional import rotate, resize, normalize
from math import degrees
import torch
import json
import imageio_ffmpeg
import torch.utils.data


rgb_mean = [95.1683, 99.7394, 98.8952]
rgb_std = [47.6498, 46.6025, 51.7941]


def reverse_norm(img):
    return img * torch.tensor(rgb_std).reshape(3, 1, 1) + torch.tensor(rgb_mean).reshape(3, 1, 1)


def make_mask(image_size):
    x = torch.linspace(-image_size // 2, image_size // 2, image_size)
    x, y = torch.meshgrid([x, x], indexing='ij')
    return torch.sqrt(x ** 2 + y ** 2).lt(image_size // 2)


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
        # x = torch.linspace(-self.image_size // 2, self.image_size // 2, self.image_size)
        # x, y = torch.meshgrid([x, x], indexing='ij')
        # self.mask = torch.sqrt(x ** 2 + y ** 2).lt(self.image_size // 2)
        self.mask = make_mask(image_size)

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
        img = normalize(img, rgb_mean, rgb_std, inplace=True)

        # load the serialized form into a numpy array
        lines = np.stack([Line.from_flat(line).to_numpy() for line in self.lines[self.index[item]]], axis=-1)
        lines[0, :, :] = lines[0, :, :] * rescale_x_factor
        lines[:, :, :] = lines[:, :, :] / scale

        # compute l2 norm for each line ( rise - run in normal form ), then convert to complex plane, and angle
        slope = (lines[:, 1, :] - lines[:, 0, :])
        length = np.linalg.norm(slope, ord=2, axis=0, keepdims=True)
        complex = (slope / length).T.copy().view(np.complex128)
        complex_mean = complex.mean()
        angle = torch.tensor([np.angle(complex_mean)])

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


def video_stream(data_dir, image_size):
    reader = imageio_ffmpeg.read_frames(f'{data_dir}/video.mp4')
    frameinfo = next(reader)
    w, h = frameinfo['size']
    mask = make_mask(image_size)
    for frame in reader:
        img = torch.frombuffer(frame[0:w * h * 3], dtype=torch.uint8)
        img = img.reshape(h, w, 3).permute(2, 0, 1)
        img = resize(img, [image_size, image_size]).float()
        img = normalize(img, rgb_mean, rgb_std, inplace=True)
        #img = img * mask
        yield img


class HorizonVideoDataset(torch.utils.data.IterableDataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, data_dir, image_size):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size

    def __iter__(self):
        return video_stream(self.data_dir, self.image_size)


if __name__ == '__main__':

    """
    run to compute the normalization values of the dataset
    """

    nimages = 0
    mean = torch.zeros(3)
    var = torch.zeros(3)

    import pathlib

    path = pathlib.Path('./data/horizon').glob('*.png')

    for file in path:
        img = read_image(str(file))
        # Rearrange batch to be the shape of [C, W * H]
        img = img.view(img.size(0), -1).float()
        # Update total number of images
        nimages += 1
        # Compute mean and std here
        mean += img.mean(1)
        var += img.var(1)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print('mean', mean)
    print('std', std)
