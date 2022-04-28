import sys
import traceback
import argparse

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject  # noqa:F401,F402


# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)

DEFAULT_PIPELINE = "videotestsrc num-buffers=100 ! autovideosink"

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

args = vars(ap.parse_args())


def on_message(bus: Gst.Bus, message: Gst.Message, loop: GObject.MainLoop):
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


command = args["pipeline"]

# Gst.Pipeline https://lazka.github.io/pgi-docs/Gst-1.0/classes/Pipeline.html
# https://lazka.github.io/pgi-docs/Gst-1.0/functions.html#Gst.parse_launch
pipeline = Gst.parse_launch(command)

# https://lazka.github.io/pgi-docs/Gst-1.0/classes/Bus.html
bus = pipeline.get_bus()

# allow bus to emit messages to main thread
bus.add_signal_watch()

# Init GObject loop to handle Gstreamer Bus Events
loop = GObject.MainLoop()

# Add handler to specific signal
# https://lazka.github.io/pgi-docs/GObject-2.0/classes/Object.html#GObject.Object.connect
bus.connect("message", on_message, loop)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)


try:
    loop.run()
except Exception:
    traceback.print_exc()
    loop.quit()

# Stop Pipeline
pipeline.set_state(Gst.State.NULL)