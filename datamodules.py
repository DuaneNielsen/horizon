from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import json
from markup_video import Line, Point
import numpy as np


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
        lines = np.stack([Line.from_flat(line).to_numpy() for line in self.lines[self.index[item]]])

        # compute l2 norm for each line ( rise - run in normal form )
        slope = (lines[:, :, 1] - lines[:, :, 0]).T
        length = np.linalg.norm(slope, axis=0, keepdims=True)
        return slope/length


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

