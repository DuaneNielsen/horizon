import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import argparse


"""
run with command 

GST_DEBUG=INFO deepstream_minimal.py --uri file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 

The equivalent pipeline is 

GST_DEBUG=4 gst-launch-1.0 uridecodebin uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 
    ! nveglglessink

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

    nveglglesssink = Gst.ElementFactory.make("nveglglessink")
    pipeline.add(nveglglesssink)

    """
    Link Elements
    """

    def uridecodebin_pad_added(uridecodebin, pad, nvvideoconvert):

        # comment out below to debug inside callbacks on pycharm
        # import pydevd_pycharm
        # pydevd_pycharm.settrace()

        print("uridecodebin_pad_added")

        caps = pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        features = caps.get_features(0)

        if "video" in gstname:
            if features.contains("memory:NVMM"):
                uridecodebin.link(nvvideoconvert)

    # dynamic linking is required for uridecodebin
    uridecodebin.connect("pad-added", uridecodebin_pad_added, nveglglesssink)

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

    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    pipeline.set_state(Gst.State.NULL)