import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import dataset
import wandb
from argparse import ArgumentParser
from dataset import HorizonDataSet
from pytorch_lightning.utilities import argparse as pl_argparse
import timm
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import random_split, dataloader
from math import ceil
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import sys
from typing import Optional, Any
import gstreamer
import cv2


class AppSrcPipeline(gstreamer.GstPipeline):
    def on_pipeline_init(self) -> None:
        # Source element for reading from the file
        print("Creating Source \n ")
        appsource = Gst.ElementFactory.make("appsrc", "numpy-source")
        if not appsource:
            sys.stderr.write(" Unable to create Source \n")

        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nv-videoconv")
        if not nvvideoconvert:
            sys.stderr.write(" error nvvid1")

        caps_filter = Gst.ElementFactory.make("capsfilter", "capsfilter1")
        if not caps_filter:
            sys.stderr.write(" error capsf1")

        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

        appsource_caps = Gst.Caps.from_string("video/x-raw,format=RGBA,width=64,height=64, framerate=1/1")
        appsource.set_property("block", True)
        appsource.set_property('caps', appsource_caps)

        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12,width=64,height=64, framerate=1/1")
        caps_filter.set_property('caps', caps)

        print("Adding elements to Pipeline \n")
        self.pipeline.add(appsource)
        self.pipeline.add(nvvideoconvert)
        self.pipeline.add(caps_filter)
        self.pipeline.add(sink)

        # Working Link pipeline
        print("Linking elements in the Pipeline \n")
        appsource.link(nvvideoconvert)
        nvvideoconvert.link(caps_filter)
        caps_filter.link(sink)


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
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)
        trainer.logger.experiment.log({
            "val/examples": [
                wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                for x, pred, y in zip(val_imgs, preds, val_labels)
            ],
            "global_step": trainer.global_step
        })


class HorizonRollClassifier(pl.LightningModule):

    def __init__(self, model_str: str, data_dir: str, lr: float = 1e-4, batch_size: int = 8,
                 select_label: str = 'discrete',
                 num_classes: int = 16, num_workers: int = 0, image_size: int = 64, no_rotate: bool = False, no_mask: bool = False,):
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        # init args
        self.dataset_kwargs = {'data_dir': data_dir, 'no_rotate': no_rotate, 'no_mask': no_mask,
                               'num_classes': num_classes, 'select_label': select_label, 'image_size': image_size}

        self.dataloader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers}
        self.model = timm.create_model(model_name=model_str, num_classes=num_classes)
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
        return torch.log_softmax(self.model.forward(x), dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss.item()}
        return {'loss': loss, 'preds': logits.detach(), 'targets': y.detach(), 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss, 'preds': logits.detach(), 'targets': y.detach()}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'test_loss': loss, 'preds': logits.detach(), 'targets': y.detach()}

    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['targets'])

    def training_epoch_end(self, outputs):
        self.log("train/acc_epoch", self.train_acc)

    def validation_step_end(self, outputs):
        self.val_acc(outputs['preds'], outputs['targets'])

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {
            'val/val_loss': avg_loss,
            'val/acc_epoch': self.val_acc
            }
        [self.log(k, v) for k, v in log.items()]
        return log

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(self.val_set, **self.dataloader_kwargs)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(self.val_set, **self.dataloader_kwargs)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return dataloader.DataLoader(self.train_set, **self.dataloader_kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        global pipeline
        appsource = pipeline.get_by_name('numpy-source')
        images, labels = batch
        for img in images:
            img = dataset.reverse_norm(img).permute(1, 2, 0).byte()
            arr = img.numpy()
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)
            gst_buffer = gstreamer.ndarray_to_gst_buffer(arr)
            appsource.emit("push-buffer", gst_buffer)
        return None


global pipeline


def predict_checkpoint(args):
    with gstreamer.GstContext():
        global pipeline
        pipeline = AppSrcPipeline()
        try:
            pipeline.startup()
            model = HorizonRollClassifier.load_from_checkpoint(args.predict_checkpoint, no_mask=args.no_mask)
            trainer = pl.Trainer()
            trainer.predict(model)
        except Exception as e:
            print('Error: ', e)
        finally:
            pipeline.shutdown()


def train(args):
    checkpoint = ModelCheckpoint(dirpath=f"checkpoints/classifier/{wandb_logger.experiment.name}/",
                                 save_top_k=2,
                                 monitor="val/acc_epoch", mode='max')

    model = HorizonRollClassifier.from_argparse_args(args)

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
    model = HorizonRollClassifier.load_from_checkpoint(args.validate_checkpoint)
    trainer = pl.Trainer(callbacks=WandbImagePredCallback(num_samples=32), logger=wandb_logger)
    trainer.validate(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    pl.Trainer.add_argparse_args(parser)
    HorizonRollClassifier.add_argparse_args(parser)
    parser.add_argument('--val_samples', type=int, default=16)
    parser.add_argument('--resume_training', type=str, default=None)
    parser.add_argument('--load_from_checkpoint', type=str, default=None)
    parser.add_argument('--validate_checkpoint', type=str, default=None)
    parser.add_argument('--predict_checkpoint', type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(1234)
    wandb_logger = WandbLogger(project='horizon')

    if args.validate_checkpoint is not None:
        validate_checkpoint(args)
    elif args.predict_checkpoint is not None:
        predict_checkpoint(args)
    else:
        train(args)

