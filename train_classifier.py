import torch
import torch.nn.functional as F
import torch.nn as nn
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
from torch.utils.data import random_split, dataloader, RandomSampler
from math import ceil
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst
import sys
from typing import Optional, Any
import gstreamer
import cv2
from pydevd_pycharm import settrace
import traceback
import pyds
import time
from pathlib import Path
import ctypes
import numpy as np
from pyds import NvDsMetaType
import onnxruntime as ort
import onnx

# from cupy.cuda.runtime import hostAlloc, memcpy, freeHost


def to_numpy(l_meta):
    user_meta = pyds.NvDsUserMeta.cast(l_meta.data)
    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    array = np.array(np.ctypeslib.as_array(ptr, shape=(layer.dims.numElements,)), copy=True)
    return array


def probe_nvdsosd_pad_src_data(pad, info):
    print('entered probe_nvsink_pad_src_data')

    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list

        l_meta = frame_meta.frame_user_meta_list

        print(f'frame_number: {frame_number}')
        print(f'l_obj: {l_obj}')
        print(f'l_meta: {l_meta}')

        while l_meta is not None:
            probs = to_numpy(l_meta)
            angle = np.argmax(probs)
            print(probs)
            print(f'estimate {angle}')
            try:
                l_meta = l_meta.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

"""
Draws inference results on the overlay
"""
def draw_line(pad, info):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_user = batch_meta.batch_user_meta_list
    angles = []

    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        base_meta = pyds.NvDsBaseMeta.cast(user_meta.base_meta)

        if base_meta.meta_type == NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            array = np.array(np.ctypeslib.as_array(ptr, shape=(layer.dims.numElements,)), copy=True)
            angles.append(np.argmax(array))

        try:
            l_user = l_user.next
        except StopIteration:
            break

    counter = 0
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:

        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

        center_x, center_y = 1280 // 2, 560 // 2
        angle = angles[counter]/16 * 2 * np.pi

        display_meta.num_lines = 1
        display_meta.line_params[0].x1 = center_x
        display_meta.line_params[0].y1 = center_y
        display_meta.line_params[0].x2 = center_x + np.floor(np.cos(angle) * 200).astype(np.int)
        display_meta.line_params[0].y2 = center_y + np.floor(np.sin(angle) * 200).astype(np.int)
        display_meta.line_params[0].line_width = 4
        display_meta.line_params[0].line_color.red = 1.0
        display_meta.line_params[0].line_color.alpha = 1.0

        display_meta.num_labels = 1
        display_meta.text_params[0].display_text = f'prediction: {angles[counter]}'
        display_meta.text_params[0].x_offset = 600
        display_meta.text_params[0].y_offset = 560
        display_meta.text_params[0].font_params.font_name = 'Serif'
        display_meta.text_params[0].font_params.font_size = 40
        display_meta.text_params[0].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        display_meta.text_params[0].set_bg_clr = 1
        display_meta.text_params[0].text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
            counter += 1
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def probe_nvdspreprocess_pad_src_data(pad, info):
    # settrace()
    print('entered probe probe_nvdspreprocess_pad_src_data')

    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_user = batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        if user_meta:

            try:

                base_meta = pyds.NvDsBaseMeta.cast(user_meta.base_meta)
                print(f'base_meta.meta_type {base_meta.meta_type}')
                if base_meta.meta_type == NvDsMetaType.NVDS_PREPROCESS_BATCH_META:
                    nvds_prepro_batch_meta = pyds.GstNvDsPreProcessBatchMeta.cast(user_meta.user_meta_data)
                    nvds_tensor = pyds.NvDsPreProcessTensorMeta.cast(nvds_prepro_batch_meta.tensor_meta)
                    print(f'buffer_size {nvds_tensor.buffer_size}')
                    print(f'tensor_shape {nvds_tensor.tensor_shape}')
                    print(f'data_type {nvds_tensor.data_type}')
                    print(f'gpu_id {nvds_tensor.gpu_id}')

                    # cuda_mem_ptr = pyds.get_ptr(nvds_tensor.raw_tensor_buffer)
                    # host_mem_ptr = hostAlloc(nvds_tensor.buffer_size, 0)
                    # memcpy(host_mem_ptr, cuda_mem_ptr, nvds_tensor.buffer_size, 0)
                    # ptr = ctypes.cast(host_mem_ptr, ctypes.POINTER(ctypes.c_float))
                    # ctypes_array = np.ctypeslib.as_array(ptr, shape=nvds_tensor.tensor_shape)
                    # array = np.array(ctypes_array, copy=True)
                    # freeHost(host_mem_ptr)

                    # print('deepstream prepro')
                    # print(array.shape)
                    # print(array)

                    # rgb_array = (array[0] * 256).astype(np.uint8).transpose(1, 2, 0)
                    # cv2.imshow("baseball", rgb_array)
                    # cv2.waitKey(1)

                if base_meta.meta_type == NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    array = np.array(np.ctypeslib.as_array(ptr, shape=(layer.dims.numElements,)), copy=True)
                    print(array)

            except StopIteration:
                break

            try:
                l_user = l_user.next
            except StopIteration:
                break

    return Gst.PadProbeReturn.OK


class DeepstreamPrepro(gstreamer.GstCommandPipeline):
    def __init__(self):
        command = f'appsrc name=numpy-source block=True caps=video/x-raw,format=RGBA,width=1280,height=660,framerate=1/1 ' \
                  f'! nvvideoconvert ' \
                  f'! m.sink_0 nvstreammux name=m gpu-id=0 batch-size=1 width=1280 height=660 batched-push-timeout=4000000 nvbuf-memory-type={int(pyds.NVBUF_MEM_CUDA_DEVICE)} ' \
                  f'! nvdspreprocess name=prepro config-file= roll_classifier_prepro.txt ' \
                  f'! nvinfer name=nvinfer config-file-path= roll_classifier_pgie_old.txt raw-output-file-write=1 ' \
                  f'! nvmultistreamtiler width=1280 height=660 ! nvvideoconvert ! nvdsosd name=nvosd ! nveglglessink'
        super().__init__(command)

    def on_pipeline_init(self) -> None:
        # prepro = self.get_by_name('prepro')
        # prepro_src = prepro.get_static_pad('src')
        # prepro_src.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data)

        nvinfer = self.get_by_name('nvinfer')
        nvinfer_sink = nvinfer.get_static_pad('sink')
        nvinfer_sink.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data)
        nvinfer_src = nvinfer.get_static_pad('src')
        nvinfer_src.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data)

        nvosd = self.get_by_name('nvosd')
        nvosd_sink = nvosd.get_static_pad('sink')
        nvosd_sink.add_probe(Gst.PadProbeType.BUFFER, draw_line)

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        # cv2.destroyAllWindows()
        self._shutdown_pipeline()


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
                 num_classes: int = 16, num_workers: int = 0, image_size: int = 64,
                 no_rotate: bool = False, no_mask: bool = False, no_resize=False, no_normalize=False,
                 return_orig: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        # init args
        self.dataset_kwargs = {'data_dir': data_dir,
                               'num_classes': num_classes, 'select_label': select_label, 'image_size': image_size,
                               'no_rotate': no_rotate, 'no_mask': no_mask, 'no_resize': no_resize,
                               'no_normalize': no_normalize, 'return_orig': return_orig}

        self.dataloader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers,
                                  'pin_memory': True, }
        self.model = timm.create_model(model_name=model_str, num_classes=num_classes)
        self.train_set = None
        self.val_set = None

    @property
    def batch_size(self):
        return self.dataloader_kwargs['batch_size']

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
        milestones = [200]
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1),
                        'name': 'learning_rate'}
        return [optimizer], [lr_scheduler]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(self.val_set, **self.dataloader_kwargs)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return dataloader.DataLoader(self.val_set, **self.dataloader_kwargs)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = RandomSampler(self.train_set, replacement=True)
        return dataloader.DataLoader(self.train_set, sampler=sampler, **self.dataloader_kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        global pipeline
        global onx_sess
        appsource = pipeline.get_by_name('numpy-source')
        images, originals, labels = batch
        self.eval()
        y = self.forward(images)

        # send images in the batch 1 by 1
        for i, (img, orig, label) in enumerate(zip(images, originals, labels)):

            # convert CHW-RBG to HWC-RGB
            seq_id = batch_idx * self.batch_size + i
            arr = orig.permute(1, 2, 0).byte().numpy()

            # attach label to image
            label_img = np.zeros(shape=(100, 1280, 3), dtype=np.uint8)
            cv2.putText(img=label_img, text=f'label: {label}', org=(20, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                        color=(0, 255, 0), thickness=2)
            arr = np.concatenate((arr, label_img), axis=0)

            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)

            # push to buffer
            gst_buffer = gstreamer.ndarray_to_gst_buffer(arr)
            appsource.emit("push-buffer", gst_buffer)
            time.sleep(1.0)

            # load nvinfer input from file and verify its the same
            infer_input = np.fromfile(
                f'gstnvdsinfer_uid-01_layer-modelInput_batch-{seq_id:010}_batchsize-01.bin',
                dtype=np.float32).reshape(1, 3, 64, 64)

            # load nvinfer output from file to verify results are the same as pytorch output
            infer_output = np.fromfile(
                f'gstnvdsinfer_uid-01_layer-modelOutput_batch-{seq_id:010}_batchsize-01.bin',
                dtype=np.float32)

            print('***********************************************')
            print('PYTORCH  OUTPUT')
            print(np.argmax(y[i].numpy()))
            print('ONNX  OUTPUT')
            outputs = ort_sess.run(None, {'modelInput': img.unsqueeze(0).numpy()})
            print(np.argmax(outputs[0][0]))
            # print(outputs[1][0])
            # print(outputs[2][0])
            print('TENSORRT OUTPUT')
            print(np.argmax(infer_output))
            print('***********************************************')

            # torch_output = self.model.forward(img.unsqueeze(0))
            # print(torch_output)
        return None


global pipeline


class DummyPrimaryModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


# Function to Convert to ONNX
def convert_ONNX(args):

    model = HorizonRollClassifier.load_from_checkpoint(args.convert_onnx_checkpoint, no_mask=True, no_resize=True,
                                                       no_normalize=True, no_rotate=True,
                                                       return_orig=True)

    checkpt_path = Path(args.convert_onnx_checkpoint)
    onnx_file = checkpt_path.parent / Path(checkpt_path.stem + '.onnx')

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, model.hparams.image_size, model.hparams.image_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      str(onnx_file),  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'])  # the model's output names
    # dynamic_axes={'modelInput': {1: 'batch_size'},  # variable length axes
    #               'modelOutput': {1: 'batch_size'}})
    print("*************************************")
    print(f'Saving onnx model to {onnx_file}')
    print("*************************************")


global ort_sess


def predict_checkpoint(args):
    global ort_sess

    model = HorizonRollClassifier.load_from_checkpoint(args.predict_checkpoint, no_mask=True, no_resize=True,
                                                       no_normalize=True, no_rotate=True,
                                                       return_orig=True)

    ort_sess = ort.InferenceSession('study/test.onnx', '', providers=['CUDAExecutionProvider'])

    with gstreamer.GstContext():
        global pipeline
        pipeline = DeepstreamPrepro()
        try:
            pipeline.startup()
            trainer = pl.Trainer()
            trainer.predict(model)
        except Exception as e:
            traceback.print_exc()
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
    parser.add_argument('--convert_onnx_checkpoint', type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(1234)
    wandb_logger = WandbLogger(project='horizon')

    if args.validate_checkpoint is not None:
        validate_checkpoint(args)
    elif args.predict_checkpoint is not None:
        predict_checkpoint(args)
    elif args.convert_onnx_checkpoint is not None:
        convert_ONNX(args)
    else:
        train(args)
