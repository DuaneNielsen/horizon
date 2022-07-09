from gstreamer import GstCommandPipeline
import time
from gi.repository import Gst
import cv2
import pyds
import ctypes
import numpy as np

# THE ORDER IN WHICH THIS IS MODULE IS INCLUDED MATTERS
from cupy.cuda.runtime import hostAlloc, memcpy, freeHost


def probe_nvdspreprocess_pad_src_data(pad, info):
    print('entered probe')

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

                print(nvds_tensor.buffer_size)
                print(nvds_tensor.tensor_shape)
                print(nvds_tensor.data_type)
                print(nvds_tensor.gpu_id)
                print(nvds_tensor.raw_tensor_buffer)

                cuda_mem_ptr = pyds.get_ptr(nvds_tensor.raw_tensor_buffer)
                host_mem_ptr = hostAlloc(nvds_tensor.buffer_size, 0)
                memcpy(host_mem_ptr, cuda_mem_ptr, nvds_tensor.buffer_size, 0)
                ptr = ctypes.cast(host_mem_ptr, ctypes.POINTER(ctypes.c_float))
                ctypes_array = np.ctypeslib.as_array(ptr, shape=nvds_tensor.tensor_shape)
                array = np.array(ctypes_array, copy=True)
                print(array.sum())
                freeHost(host_mem_ptr)

                rgb_array = (array[0] * 256).astype(np.uint8).transpose(1, 2, 0)
                cv2.imshow("baseball", rgb_array)
                cv2.waitKey(1)

            except StopIteration:
                break

            try:
                l_user = l_user.next
            except StopIteration:
                break

    return Gst.PadProbeReturn.OK


class DeepstreamPreProcessor(GstCommandPipeline):
    def __init__(self):
        super().__init__(
            f'multifilesrc location = data/horizon/frame_%05d.png index=0 '
            f'caps=image/png,framerate=(fraction)12/1 '
            f'! pngdec '
            f'! videoconvert '
            f'! videorate '
            f'! nvvideoconvert '
            f'! m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=560 nvbuf-memory-type=2 '
            f'! nvdspreprocess config-file=roll_classifier_prepro.txt '
            f'! nvvideoconvert '
            f'! nveglglessink'
        )

    def on_pipeline_init(self) -> None:
        super().on_pipeline_init()

        prepro = self.get_by_name('nvdspreprocess0')
        prepro_src = prepro.get_static_pad('src')
        prepro_src.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data)


if __name__ == '__main__':

    with DeepstreamPreProcessor() as pipeline:
        while not pipeline.is_done:
            time.sleep(.1)
