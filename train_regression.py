import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from argparse import ArgumentParser
from dataset import HorizonDataSet
from pytorch_lightning.utilities import argparse as pl_argparse
import timm
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, dataloader
from math import ceil, degrees
import wandb


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
        trainer.logger.experiment.log({
            "val/examples": [
                wandb.Image(x,
                            caption=f"Pred: {degrees(pred.angle().item()):.1f}\u00B0, "
                                    f"Label: {degrees(y.angle().item()):.1f}\u00B0"
                            )
                for x, pred, y in zip(val_imgs, preds, val_labels)
            ],
        })


class HorizonRollRegression(pl.LightningModule):

    def __init__(self, model_str: str, data_dir: str, lr: float = 1e-4, gamma: float = 0.99, batch_size: int = 8,
                 select_label: str = 'complex_mean', num_workers: int = 0, image_size: int = 64, rotate: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

        self.val_loss = torchmetrics.MeanMetric()
        self.train_loss = torchmetrics.MeanMetric()

        # init args
        self.dataset_kwargs = {'data_dir': data_dir, 'rotate': rotate, 'num_classes': 2,
                               'select_label': select_label, 'image_size': image_size}

        self.dataloader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers}
        self.model = timm.create_model(self.hparams.model_str, num_classes=2)
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
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)
    HorizonRollRegression.add_argparse_args(parser)
    parser.add_argument('--val_samples', type=int, default=16)
    args = parser.parse_args()

    pl.seed_everything(1234)

    wandb_logger = WandbLogger(project='horizon_regression')
    checkpoint = ModelCheckpoint(dirpath=f"checkpoints/{wandb_logger.experiment.name}/",
                                 save_top_k=2,
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

    trainer.fit(model)
