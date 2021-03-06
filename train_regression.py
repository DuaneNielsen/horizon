from typing import Optional, Any
from math import ceil, floor, degrees
from argparse import ArgumentParser

import torch
from torch.utils.data import random_split, dataloader
from torchvision.transforms.functional import to_pil_image
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.utilities import argparse as pl_argparse

import timm
import wandb
from PIL import ImageDraw
import cv2

from dataset import HorizonDataSet, HorizonVideoDataset, reverse_norm


class WandbImagePredCallback(pl.Callback):
    """Logs the input images and output predictions of a module.
    Predictions and labels are logged as class indices."""

    def __init__(self, num_samples=32):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = pl_module.val_dataloader()
        val_data = []
        for i in range(self.num_samples):
            val_data += [val_loader.dataset[i]]
        val_data = list(zip(*val_data))
        val_imgs = torch.stack(val_data[0]).to(device=pl_module.device)
        val_labels = torch.tensor(val_data[1]).to(device=pl_module.device)
        preds = pl_module(val_imgs)

        def draw_horizon(img, pred, label):
            C, H, W = img.shape
            c_x, c_y = W // 2, H // 2
            img = reverse_norm(img).byte()
            img = to_pil_image(img)
            img1 = ImageDraw.Draw(img)
            img1.line([(c_x, c_y), (pred.real * c_x + c_x, pred.imag * c_y + c_y)], fill="red", width=1)
            img1.line([(c_x, c_y), (label.real * c_x + c_x, label.imag * c_y + c_y)], fill="blue", width=1)
            return img

        trainer.logger.experiment.log({
            "val/examples": [
                wandb.Image(draw_horizon(x, pred, y),
                            caption=f"Pred: {degrees(pred.angle().item()):.1f}\u00B0, "
                                    f"Label: {degrees(y.angle().item()):.1f}\u00B0"
                            )
                for x, pred, y in zip(val_imgs, preds, val_labels)
            ],
        })


class HorizonRollRegression(pl.LightningModule):

    def __init__(self, model_str: str, data_dir: str, lr: float = 1e-4, gamma: float = 0.99, batch_size: int = 8,
                 select_label: str = 'complex_mean', num_workers: int = 0, image_size: int = 64,
                 no_rotate: bool = False, no_mask: bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

        self.val_loss = torchmetrics.MeanMetric()
        self.train_loss = torchmetrics.MeanMetric()

        # init args
        self.dataset_kwargs = {'data_dir': data_dir, 'no_rotate': no_rotate, 'num_classes': 2,
                               'select_label': select_label, 'image_size': image_size, 'no_mask': no_mask}

        self.dataloader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers}
        self.model = timm.create_model(model_name=self.hparams.model_str, num_classes=2)
        self.train_set = None
        self.val_set = None

    @classmethod
    def from_argparse_args(cls, args):
        kwargs = {}
        for name, tipe, default in pl_argparse.get_init_arguments_and_types(cls):
            kwargs[name] = vars(args)[name]
        return cls(**kwargs)

    @classmethod
    def add_argparse_args(cls, parser):
        for name, tipe, default in pl_argparse.get_init_arguments_and_types(cls):
            if tipe[0] == bool:
                parser.add_argument(f'--{name}', action='store_true')
            else:
                parser.add_argument(f'--{name}', type=tipe[0], default=default)

    def setup(self, stage=None):
        full = HorizonDataSet(**self.dataset_kwargs)
        train_size, val_size = len(full) * 9 // 10, ceil(len(full) / 10)
        self.train_set, self.val_set = random_split(full, [train_size, val_size])

    def forward(self, x):
        out = torch.view_as_complex(self.model.forward(x))
        return (out / out.abs()).unsqueeze(1)

    def abs_loss(self, pred, target):
        return (pred - target).abs().pow(2).mean()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.forward(x)
        loss = self.abs_loss(pred, y)

        logs = {'train_loss': loss.item()}
        return {'loss': loss, 'preds': pred.detach(), 'targets': y.detach(), 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.forward(x)
        loss = self.abs_loss(pred, y)

        return {'loss': loss, 'preds': pred.detach(), 'targets': y.detach()}

    def test_step(self, val_batch, batch_idx):
        pass

    def training_step_end(self, outputs):
        self.train_mae(outputs['preds'].angle() * 180.0 / torch.pi, outputs['targets'].angle() * 180.0 / torch.pi)
        self.train_loss(outputs['loss'])

    def training_epoch_end(self, outputs):
        self.log('train/loss', self.train_loss)
        self.log("train/mean_abs_error_epoch (degrees)", self.train_mae)

    def validation_step_end(self, outputs):
        self.val_mae(outputs['preds'].angle() * 180.0 / torch.pi, outputs['targets'].angle() * 180.0 / torch.pi)
        self.val_loss(outputs['loss'])

    def validation_epoch_end(self, outputs):
        log = {
            'val/val_loss': self.val_loss,
            'val/mean_abs_error_epoch (degrees)': self.val_mae
        }
        [self.log(k, v) for k, v in log.items()]
        return log

    def test_step_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return dataloader.DataLoader(self.train_set, **self.dataloader_kwargs)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(self.val_set, **self.dataloader_kwargs)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(HorizonVideoDataset(**self.dataset_kwargs))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        predict = self.forward(batch)
        return {'predict': predict }

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pred = outputs['predict'].squeeze()
        img = reverse_norm(batch.squeeze())
        img = img.permute(1, 2, 0).byte().numpy()
        img = cv2.resize(img, (512, 512))
        x, y = floor(pred.real.item() * 256 + 256), floor(pred.imag.item() * 256 + 256)
        img = cv2.line(img, pt1=(256, 256), pt2=(x, y), color=(0, 0, 255), thickness=3)
        cv2.imshow('inference', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()


def train(args):
    checkpoint = ModelCheckpoint(dirpath=f"checkpoints/regression/{wandb_logger.experiment.name}/",
                                 save_top_k=2, mode='min',
                                 monitor="val/val_loss")

    model = HorizonRollRegression.from_argparse_args(args)

    trainer = pl.Trainer.from_argparse_args(args,
                                            strategy=DDPPlugin(find_unused_parameters=False),
                                            callbacks=[
                                                LearningRateMonitor(),
                                                WandbImagePredCallback(num_samples=args.val_samples),
                                                checkpoint
                                            ],
                                            enable_checkpointing=True,
                                            default_root_dir='.',
                                            logger=wandb_logger)

    trainer.fit(model, ckpt_path=args.resume_training)


def validate_checkpoint(args):
    model = HorizonRollRegression.load_from_checkpoint(args.validate_checkpoint)
    trainer = pl.Trainer(callbacks=WandbImagePredCallback(num_samples=32), logger=wandb_logger)
    trainer.validate(model)


def predict_checkpoint(args):
    model = HorizonRollRegression.load_from_checkpoint(args.predict_checkpoint, no_mask=args.no_mask)
    trainer = pl.Trainer()
    trainer.predict(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)
    HorizonRollRegression.add_argparse_args(parser)
    parser.add_argument('--val_samples', type=int, default=16)
    parser.add_argument('--resume_training', type=str, default=None)
    parser.add_argument('--load_from_checkpoint', type=str, default=None)
    parser.add_argument('--validate_checkpoint', type=str, default=None)
    parser.add_argument('--predict_checkpoint', type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(1234)
    wandb_logger = WandbLogger(project='horizon_regression')

    if args.validate_checkpoint is not None:
        validate_checkpoint(args)
    elif args.predict_checkpoint is not None:
        predict_checkpoint(args)
    else:
        train(args)
