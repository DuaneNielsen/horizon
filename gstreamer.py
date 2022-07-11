import logging
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import GObject, Gst, GLib
import numpy as np
import threading
import typing as typ
import sys

Gst.init(sys.argv if hasattr(sys, "argv") else None)


def ndarray_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:
    """Converts numpy array to Gst.Buffer"""
    return Gst.Buffer.new_wrapped(array.tobytes())


class GstContext:
    def __init__(self):
        # SIGINT handle issue:
        # https://github.com/beetbox/audioread/issues/63#issuecomment-390394735
        self._main_loop = GLib.MainLoop.new(None, False)

        self._main_loop_thread = threading.Thread(target=self._main_loop_run)

        self._log = logging.getLogger("pygst.{}".format(self.__class__.__name__))

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def log(self) -> logging.Logger:
        return self._log

    def startup(self):
        if self._main_loop_thread.is_alive():
            return

        self._main_loop_thread.start()

    def _main_loop_run(self):
        try:
            self._main_loop.run()
        except Exception:
            pass

    def shutdown(self, timeout: int = 2):
        self.log.debug("%s Quitting main loop ...", self)

        if self._main_loop.is_running():
            self._main_loop.quit()

        self.log.debug("%s Joining main loop thread...", self)
        try:
            if self._main_loop_thread.is_alive():
                self._main_loop_thread.join(timeout=timeout)
        except Exception as err:
            self.log.error("%s.main_loop_thread : %s", self, err)
            pass


def gst_state_to_str(state: Gst.State) -> str:
    """Converts Gst.State to str representation
    Explained: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Element.html#Gst.Element.state_get_name
    """
    return Gst.Element.state_get_name(state)


class GstPipeline:
    """Base class to initialize any Gstreamer Pipeline from string"""

    def __init__(self):
        """
        :param command: gst-launch string
        """
        #self._command = command
        self._pipeline = None  # Gst.Pipeline
        self._bus = None  # Gst.Bus

        self._log = logging.getLogger("pygst.{}".format(self.__class__.__name__))
        self._end_stream_event = threading.Event()

    @property
    def log(self) -> logging.Logger:
        return self._log

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return "<{}>".format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def get_by_cls(self, cls: GObject.GType) -> typ.List[Gst.Element]:
        """ Get Gst.Element[] from pipeline by GType """
        elements = self._pipeline.iterate_elements()
        if isinstance(elements, Gst.Iterator):
            # Patch "TypeError: ‘Iterator’ object is not iterable."
            # For versions we have to get a python iterable object from Gst iterator
            _elements = []
            while True:
                ret, el = elements.next()
                if ret == Gst.IteratorResult(1):  # GST_ITERATOR_OK
                    _elements.append(el)
                else:
                    break
            elements = _elements

        return [e for e in elements if isinstance(e, cls)]

    def get_by_name(self, name: str) -> Gst.Element:
        """Get Gst.Element from pipeline by name
        :param name: plugins name (name={} in gst-launch string)
        """
        return self._pipeline.get_by_name(name)

    def init_pipeline(self):
        GObject.threads_init()
        Gst.init(None)
        self._pipeline = Gst.Pipeline()

    def startup(self):
        """ Starts pipeline """
        if self._pipeline:
            raise RuntimeError("Can't initiate %s. Already started")

        self.init_pipeline()

        # Initialize Bus
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self.bus.connect("message::error", self.on_error)
        self.bus.connect("message::eos", self.on_eos)
        self.bus.connect("message::warning", self.on_warning)

        # Initalize Pipeline
        self.on_pipeline_init()
        self._pipeline.set_state(Gst.State.READY)

        self.log.info("Starting %s", self)

        self._end_stream_event.clear()

        self.log.debug(
            "%s Setting pipeline state to %s ... ",
            self,
            gst_state_to_str(Gst.State.PLAYING),
        )
        self._pipeline.set_state(Gst.State.PLAYING)
        self.log.debug(
            "%s Pipeline state set to %s ", self, gst_state_to_str(Gst.State.PLAYING)
        )

    def on_pipeline_init(self) -> None:
        """Sets additional properties for plugins in Pipeline"""
        pass

    @property
    def bus(self) -> Gst.Bus:
        return self._bus

    @property
    def pipeline(self) -> Gst.Pipeline:
        return self._pipeline

    def _shutdown_pipeline(self, timeout: int = 1, eos: bool = False) -> None:
        print('SHUTDOWN_PIPELINE')
        """ Stops pipeline
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """

        if self._end_stream_event.is_set():
            return

        self._end_stream_event.set()

        if not self.pipeline:
            return

        self.log.debug("%s Stopping pipeline ...", self)

        # https://lazka.github.io/pgi-docs/Gst-1.0/classes/Element.html#Gst.Element.get_state
        if self._pipeline.get_state(timeout=1)[1] == Gst.State.PLAYING:
            self.log.debug("%s Sending EOS event ...", self)
            try:
                thread = threading.Thread(
                    target=self._pipeline.send_event, args=(Gst.Event.new_eos(),)
                )
                thread.start()
                thread.join(timeout=timeout)
            except Exception:
                pass

        self.log.debug("%s Reseting pipeline state ....", self)
        try:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        except Exception:
            pass

        self.log.debug("%s Gst.Pipeline successfully destroyed", self)

    def shutdown(self, timeout: int = 1, eos: bool = False) -> None:
        """Shutdown pipeline
        :param timeout: time to wait when pipeline fully stops
        :param eos: if True -> send EOS event
            - EOS event necessary for FILESINK finishes properly
            - Use when pipeline crushes
        """
        self.log.info("%s Shutdown requested ...", self)

        self._shutdown_pipeline(timeout=timeout, eos=eos)

        self.log.info("%s successfully destroyed", self)

    @property
    def is_active(self) -> bool:
        return self.pipeline is not None and not self.is_done

    @property
    def is_done(self) -> bool:
        return self._end_stream_event.is_set()

    def on_error(self, bus: Gst.Bus, message: Gst.Message):
        err, debug = message.parse_error()
        self.log.error("Gstreamer.%s: Error %s: %s. ", self, err, debug)
        self._shutdown_pipeline()

    def on_eos(self, bus: Gst.Bus, message: Gst.Message):
        self.log.debug("Gstreamer.%s: Received stream EOS event", self)
        self._shutdown_pipeline()

    def on_warning(self, bus: Gst.Bus, message: Gst.Message):
        warn, debug = message.parse_warning()
        self.log.warning("Gstreamer.%s: %s. %s", self, warn, debug)


class GstCommandPipeline(GstPipeline):
    def __init__(self, command):
        super().__init__()
        self._command = command
        self._log.info("%s \n gst-launch-1.0 %s", self, command)

    def init_pipeline(self):
        self._pipeline = Gst.parse_launch(self._command)

    def on_pipeline_init(self) -> None:
        """Sets additional properties for plugins in Pipeline"""
        pass
