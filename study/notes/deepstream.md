install deepstream from [these instructions](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

verify by running the below

```bash
GST_DEBUG=3 gst-launch-1.0 videotestsrc num-buffers=50 ! autovideosink
```

```bash
deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

```bash
GST_DEBUG=3 gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! avdec_h264 ! nvvideoconvert ! nveglglessink
```
=======
you will need the rtp server

```
sudo apt-get install libgstrtspserver-1.0
```

also the python bindings will not be installed, install the python samples, they include instructions on how to install the python bindings

https://github.com/NVIDIA-AI-IOT/deepstream_python_apps


command line to test loading files...

```
gst-launch-1.0 filesrc location=${FILE} ! h264parse ! decodebin ! videoconvert ! autovideosink
```

the below will need to be in your .bashrc

```
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/lib/gstreamer-1.0:$LD_LIBRARY_PATH
export LD_RUN_PATH=/usr/local/lib/gstreamer-1.0:$LD_RUN_PATH

```

and you will probably find some completion helpful too
if so, put below in .bashrc

```
NVDS_TOKENS=`gst-inspect-1.0 | grep ^nv | cut -f3 -d' ' | cut -f1 -d: | xargs`
GST_TOKENS='uridecode'
COMPLETE_TOKENS="${GST_TOKENS} ${NVDS_TOKENS}"
echo $COMPLETE_TOKENS
complete -W '${COMPLETE_TOKENS}' gst-launch-1.0
complete -W '${COMPLETE_TOKENS}' gst-inspect-1.0
```