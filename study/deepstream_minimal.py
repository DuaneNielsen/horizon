import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import argparse
import sys

"""
run with command 

GST_DEBUG=INFO deepstream_minimal.py --input /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 

The pipeline is 

GST_DEBUG=3 gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! 
    qtdemux ! h264parse ! avdec_h264 ! nvvideoconvert ! nveglglessink

"""


class Message:
    def __init__(self, message):
        self.ev = message
        self.type = message.type

    def __repr__(self):
        return f"{self.type}"

    @staticmethod
    def make(message):
        if message.type == Gst.MessageType.STATE_CHANGED:
            return StateChanged(message)
        return Message(message)


class StateChanged(Message):
    def __init__(self, message):
        super().__init__(message)


class Bus:
    def __init__(self, bus):
        self.bus = bus
        self.name = bus.name

    def __repr__(self):
        return f'{self.name}'

    @staticmethod
    def make(bus):
        return Bus(bus)


def demuxer_pad_added(demuxer, pad, data):
    print('***********************************')
    print(demuxer, pad, data)
    print('***********************************')


def on_message(bus, message, pipeline):
    sys.stdout.write(f"bus_call {message.type} {pipeline}\n")
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        Object.set_property("drop-on-latency", True)


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    args = parser.parse_args()

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()

    # filesrc = Gst.ElementFactory.make("filesrc", "filesrc")
    # filesrc.set_property('location', args.input)

    # qtdemux = Gst.ElementFactory.make("qtdemux", 'qtdemux')
    # qtdemux.connect("pad-added", demuxer_pad_added)
    #
    # h264parse = Gst.ElementFactory.make("h264parse", "h256parse")
    #
    # avdec_h264 = Gst.ElementFactory.make("avdec_h264", "avdec_h264")

    uridecoder = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    uridecoder.connect("pad-added", cb_newpad)
    uridecoder.connect("child-added", decodebin_child_added)

    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")

    nveglglesssink = Gst.ElementFactory.make("nveglglessink", "nveglglessink")

    """
    Link
    """

    # filesrc.link(qtdemux)
    # qtdemux.link(h264parse)
    # h264parse.link(avdec_h264)
    # avdec_h264.link(nvvideoconvert)
    nvvideoconvert.link(nveglglesssink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.enable_sync_message_emission()


    loop = GObject.MainLoop()
    bus.connect("message", on_message, loop)
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)