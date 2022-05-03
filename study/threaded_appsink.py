import cv2

import sys

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp
import numpy as np
import time
import gstreamer


class AppSrcPipeLine(gstreamer.GstPipeline):
    def on_pipeline_init(self) -> None:
        # Source element for reading from the file
        print("Creating Source \n ")
        appsource = Gst.ElementFactory.make("appsrc", "numpy-source")
        if not appsource:
            sys.stderr.write(" Unable to create Source \n")

        # instructs appsrc that we will be dealing with timed buffer
        appsource.set_property("format", Gst.Format.TIME)

        # instructs appsrc to block pushing buffers until ones in queue are preprocessed
        # allows to avoid huge queue internal queue size in appsrc
        appsource.set_property("block", True)

        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nv-videoconv")
        if not nvvideoconvert:
            sys.stderr.write(" error nvvid1")

        caps_filter = Gst.ElementFactory.make("capsfilter", "capsfilter1")
        if not caps_filter:
            sys.stderr.write(" error capsf1")

        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

        caps_in = Gst.Caps.from_string("video/x-raw,format=RGBA,width=640,height=480,framerate=30/1")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1")
        appsource.set_property('caps', caps_in)
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


def main():

    print("Creating Pipeline \n ")

    with gstreamer.GstContext():
        pipeline = AppSrcPipeLine()

        try:
            pipeline.startup()
            appsource = pipeline.get_by_name('numpy-source')

            # Push buffer and check
            pts = 0
            duration = 10 ** 9 / ( 1 / 1)
            for _ in range(10):
                arr = np.random.randint(low=0, high=255, size=(480, 640, 3), dtype=np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
                gst_buffer = gstreamer.ndarray_to_gst_buffer(arr)
                pts += duration  # Increase pts by duration
                gst_buffer.pts = pts
                gst_buffer.duration = duration
                appsource.emit("push-buffer", gst_buffer)
                print('push-buffer')
            appsource.emit("end-of-stream")

            while not pipeline.is_done:
                time.sleep(0.1)
        except Exception as e:
            print('Error: ', e)
        finally:
            pipeline.shutdown()


if __name__ == '__main__':
    sys.exit(main())