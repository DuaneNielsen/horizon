from argparse import ArgumentParser
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import gstreamer
import pyds
import ctypes
import numpy as np
import traceback
import time



def draw_line(pad, info):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_user = batch_meta.batch_user_meta_list
    angles = []

    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        base_meta = pyds.NvDsBaseMeta.cast(user_meta.base_meta)

        if base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
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
        display_meta.text_params[0].x_offset = 20
        display_meta.text_params[0].y_offset = 20
        display_meta.text_params[0].font_params.font_name = 'Serif'
        display_meta.text_params[0].font_params.font_size = 20
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



class DeepstreamPrepro(gstreamer.GstCommandPipeline):
    def __init__(self, filename, rtsp_url=None):
        if rtsp_url:
            sink = f'nvvideoconvert ! x264enc speed-preset=veryfast tune=zerolatency bitrate=800 ! rtspclientsink location={rtsp_url}'
        else:
            sink = 'nveglglessink'

        command = f'filesrc location={filename} ! qtdemux ! h264parse ! avdec_h264 ' \
                  f'! nvvideoconvert ' \
                  f'! m.sink_0 nvstreammux name=m gpu-id=0 batch-size=1 width=1280 height=560 batched-push-timeout=4000000 nvbuf-memory-type={int(pyds.NVBUF_MEM_CUDA_DEVICE)} ' \
                  f'! nvdspreprocess name=prepro config-file= roll_classifier_prepro_infer.txt ' \
                  f'! nvinfer name=nvinfer config-file-path= roll_classifier_pgie_old.txt ' \
                  f'! nvmultistreamtiler width=1280 height=560 ! nvvideoconvert ! nvdsosd name=nvosd ' \
                  f'! {sink}'
        print(command)
        super().__init__(command)

    def on_pipeline_init(self) -> None:

        nvosd = self.get_by_name('nvosd')
        nvosd_sink = nvosd.get_static_pad('sink')
        nvosd_sink.add_probe(Gst.PadProbeType.BUFFER, draw_line)

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        self._shutdown_pipeline()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--rtsp-url')
    args = parser.parse_args()
    pipeline = DeepstreamPrepro(args.filename, rtsp_url=args.rtsp_url)

    try:
        pipeline.startup()
        while not pipeline.is_done:
            time.sleep(.1)

    except Exception as e:
        traceback.print_exc()
    finally:
        pipeline.shutdown()