from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import json
from markup_video import Line, Point
import numpy as np
from torchvision.io import read_image


class HorizonDataSet:
    def __init__(self, data_dir='data/horizon'):
        self.data_dir = data_dir
        with open(f'{self.data_dir}/lines.json') as f:
            lines = json.loads(f.read())
        self.lines = lines
        self.index = list(lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        # load the serialized form into a numpy array
        lines = np.stack([Line.from_flat(line).to_numpy() for line in self.lines[self.index[item]]], axis=-1)

        # compute l2 norm for each line ( rise - run in normal form ), then convert to complex plane, and angle
        slope = (lines[:, 1, :] - lines[:, 0, :])
        length = np.linalg.norm(slope, ord=2, axis=0, keepdims=True)
        complex = (slope / length).T.copy().view(np.complex128)
        angle = np.angle(complex.mean())
        labels = {
            'lines': lines,
            'complex': complex,
            'angle': angle
        }
        return read_image(f'{self.data_dir}/{self.index[item]}.png'), labels


class HorizonAnglesDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

