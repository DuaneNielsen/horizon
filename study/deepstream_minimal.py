import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import argparse


"""
run with command 

GST_DEBUG=INFO deepstream_minimal.py --input /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 

The pipeline is 

GST_DEBUG=3 gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! 
    qtdemux ! h264parse ! avdec_h264 ! nvvideoconvert ! nveglglessink

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    args = parser.parse_args()

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()

    filesrc = Gst.ElementFactory.make("filesrc", "filesrc")
    filesrc.set_property('location', args.input)
    pipeline.add(filesrc)

    qtdemux = Gst.ElementFactory.make("qtdemux", 'qtdemux')
    pipeline.add(qtdemux)

    avdec_h264 = Gst.ElementFactory.make("avdec_h264", "avdec_h264")
    pipeline.add(avdec_h264)

    def demuxer_pad_added(demuxer, pad, avdec_h264):
        # import pydevd_pycharm
        # pydevd_pycharm.settrace()
        if pad.name == 'video_0':
            demuxer.link(avdec_h264)

    qtdemux.connect("pad-added", demuxer_pad_added, avdec_h264)

    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")
    pipeline.add(nvvideoconvert)

    nveglglesssink = Gst.ElementFactory.make("nveglglessink", "nveglglessink")
    pipeline.add(nveglglesssink)

    """
    Link
    """

    filesrc.link(qtdemux)
    avdec_h264.link(nvvideoconvert)
    nvvideoconvert.link(nveglglesssink)

    loop = GObject.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)