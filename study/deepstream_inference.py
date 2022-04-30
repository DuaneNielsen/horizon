import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import argparse


"""
run with command 

GST_DEBUG=INFO deepstream_minimal.py --uri file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 

The equivalent pipeline is 

gst-launch-1.0 uridecodebin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4  \
! m.sink_0 nvstreammux name=m width=1920 height=1080 batch-size=1 batched-push-timeout=4000000 \
! nvinfer config-file-path=dstest1_pgie_config.txt \
! nvvideoconvert ! nvosdbin ! nveglglessink

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', '-i', type=str)
    args = parser.parse_args()

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()

    """
    make Elements
    """

    uridecodebin = Gst.ElementFactory.make("uridecodebin")
    uridecodebin.set_property("uri", args.uri)
    pipeline.add(uridecodebin)

    nvstreammux = Gst.ElementFactory.make("nvstreammux")
    nvstreammux.set_property('width', 1920)
    nvstreammux.set_property('height', 1080)
    nvstreammux.set_property('batch-size', 1)
    nvstreammux.set_property('batched-push-timeout', 4000000)
    nvstreammux_sink_0 = nvstreammux.get_request_pad("sink_0")
    pipeline.add(nvstreammux)

    bounding_boxes = Gst.ElementFactory.make("nvinfer")
    bounding_boxes.set_property('config-file-path', "dstest1_pgie_config.txt")
    pipeline.add(bounding_boxes)

    nvvidoeconvert = Gst.ElementFactory.make("nvvideoconvert")
    pipeline.add(nvvidoeconvert)

    nvdsosd = Gst.ElementFactory.make("nvdsosd")
    pipeline.add(nvdsosd)

    nveglglesssink = Gst.ElementFactory.make("nveglglessink")
    pipeline.add(nveglglesssink)

    """
    Link Elements
    """

    def uridecodebin_pad_added(uridecodebin, pad, nvstreammux_sink_0):

        # comment out below to debug inside callbacks on pycharm
        # import pydevd_pycharm
        # pydevd_pycharm.settrace()

        print("uridecodebin_pad_added")

        caps = pad.get_current_caps()
        gststruct = caps.get_structure(0)
        features = caps.get_features(0)

        if "video" in gststruct.get_name():
            if features.contains("memory:NVMM"):
                pad.link(nvstreammux_sink_0)

    # dynamic linking is required for uridecodebin
    uridecodebin.connect("pad-added", uridecodebin_pad_added, nvstreammux_sink_0)
    nvstreammux.link(bounding_boxes)
    bounding_boxes.link(nvvidoeconvert)
    nvvidoeconvert.link(nvdsosd)
    nvdsosd.link(nveglglesssink)

    """
    Main loop
    """

    loop = GObject.MainLoop()

    # handle window close event
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message, loop):
        if message.type == Gst.MessageType.ERROR:
            if message.parse_error()[0].message == 'Output window was closed':
                loop.quit()

    bus.connect('message', on_message, loop)

    # start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    pipeline.set_state(Gst.State.NULL)
    loop.quit()