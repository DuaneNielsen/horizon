from gstreamer import GstCommandPipeline
import time
from gi.repository import Gst
import cv2
import pyds
import ctypes
import numpy as np
from PIL import Image
import logging
import argparse
import pathlib
import shutil

logging.basicConfig(level=logging.DEBUG)
import json
import os

# THE ORDER IN WHICH THIS IS MODULE IS INCLUDED MATTERS
from cupy.cuda.runtime import hostAlloc, memcpy, freeHost

RGB_means = []

def probe_nvdspreprocess_pad_src_data(pad, info, pipeline):

    global RGB_means
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
                nvds_prepro_batch_meta = pyds.GstNvDsPreProcessBatchMeta.cast(user_meta.user_meta_data)
                nvds_tensor = pyds.NvDsPreProcessTensorMeta.cast(nvds_prepro_batch_meta.tensor_meta)

                cuda_mem_ptr = pyds.get_ptr(nvds_tensor.raw_tensor_buffer)
                host_mem_ptr = hostAlloc(nvds_tensor.buffer_size, 0)
                memcpy(host_mem_ptr, cuda_mem_ptr, nvds_tensor.buffer_size, 0)
                ptr = ctypes.cast(host_mem_ptr, ctypes.POINTER(ctypes.c_float))
                ctypes_array = np.ctypeslib.as_array(ptr, shape=nvds_tensor.tensor_shape)
                array = np.array(ctypes_array, copy=True).astype(np.float32)
                freeHost(host_mem_ptr)

                frame = pipeline.keys[pipeline.counter].split('.')[0]
                np.save(f'{pipeline.data_dir}/normalized/{frame}', array)
                pipeline.counter += 1

                RGB_means += [array.mean((0, 2, 3))]

                rgb_array = (array[0] * 256).astype(np.uint8).transpose(1, 2, 0)
                cv2.imwrite(f'{pipeline.data_dir}/normalized_jpg/{frame}.jpg', rgb_array)

            except StopIteration:
                break

            try:
                l_user = l_user.next
            except StopIteration:
                break

    return Gst.PadProbeReturn.OK


class DeepstreamPreProcessor(GstCommandPipeline):
    def __init__(self, data_dir, start_index=0):
        self.data_dir = data_dir
        self.counter = start_index
        pathlib.Path(f'{self.data_dir}/normalized').mkdir(parents=True, exist_ok=True)
        pathlib.Path(f'{self.data_dir}/normalized_jpeg').mkdir(parents=True, exist_ok=True)
        super().__init__(
            f'multifilesrc location = {self.data_dir}/frame_%05d.png start-index={start_index} '
            f'caps=image/png,framerate=(fraction)12/1 '
            f'! pngdec '
            f'! videoconvert '
            f'! videorate '
            f'! nvvideoconvert '
            f'! m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=560 nvbuf-memory-type=2 '
            f'! nvdspreprocess config-file=roll_classifier_prepro.txt enable=1 '
            f'! nvvideoconvert '
            f'! nveglglessink'
        )
        self.lines = json.load(open(f'./{self.data_dir}/lines.json', 'r'))
        self.keys = [key for key in self.lines]

    def on_pipeline_init(self) -> None:
        prepro = self.get_by_name('nvdspreprocess0')
        if prepro is not None:
            prepro_src = prepro.get_static_pad('src')
            prepro_src.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data, self)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default='data/horizon')
    args = parser.parse_args()

    """
    HACK: since nvdsprepro will drop the last 2 images, we will add 2 "dummy" images
    """

    lines = json.load(open(f'./{args.data_dir}/lines.json', 'r'))
    num_frames = len(lines)
    src_file = f'{args.data_dir}/frame_{num_frames - 1:05}.png'
    for i in range(2):
        dst_file = f'{args.data_dir}/frame_{num_frames + i:05}.png'
        shutil.copy(src_file, dst_file)

    with DeepstreamPreProcessor(args.data_dir) as pipeline:
        while not pipeline.is_done:
            time.sleep(4)
            message = pipeline.bus.timed_pop_filtered(1000, Gst.MessageType.EOS)
            if message is not None:
               break

    # cleanup dummy  files
    for i in range(2):
        dst_file = f'{args.data_dir}/frame_{num_frames + i:05}.png'
        os.remove(dst_file)

    print(RGB_means)
    print(np.stack(RGB_means).mean(axis=0))