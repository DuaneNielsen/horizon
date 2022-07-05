import logging
import time
from gstreamer import GstCommandPipeline
from gi.repository import Gst
import pyds
import ctypes
import numpy as np
from pydevd_pycharm import settrace
from cupy.cuda.runtime import hostAlloc, memcpy, freeHost
import cv2

logging.basicConfig(level='INFO')


def to_numpy(l_meta):
    user_meta = pyds.NvDsUserMeta.cast(l_meta.data)
    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    array = np.array(np.ctypeslib.as_array(ptr, shape=(layer.dims.numElements,)), copy=True)
    return array


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
                rgb_array = (array[0] * 256).astype(np.uint8).transpose(1, 2, 0)
                cv2.imshow("baseball", rgb_array)
                cv2.waitKey(1)
                freeHost(host_mem_ptr)

            except StopIteration:
                break

            try:
                l_user = l_user.next
            except StopIteration:
                break

    return Gst.PadProbeReturn.OK


class DeepstreamPrepro(GstCommandPipeline):
    def __init__(self):
        command = 'filesrc location = /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nvvideoconvert ! nvdspreprocess name=prepro config-file= roll_classifier_prepro.txt  ! nvinfer config-file-path= /opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_infer_primary.txt input-tensor-meta=1 batch-size=7  ! nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! nveglglessink'
        super().__init__(command)

    def on_pipeline_init(self) -> None:
        prepro = self.get_by_name('prepro')
        prepro_src = prepro.get_static_pad('src')
        prepro_src.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data)


if __name__ == '__main__':

    with DeepstreamPrepro() as pipeline:
        while not pipeline.is_done:
            time.sleep(.1)

        cv2.destroyAllWindows()