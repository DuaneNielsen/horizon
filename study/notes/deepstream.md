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
GST_TOKENS='uridecodebin'
COMPLETE_TOKENS="${GST_TOKENS} ${NVDS_TOKENS}"
complete -W '${COMPLETE_TOKENS}' gst-launch-1.0
complete -W '${COMPLETE_TOKENS}' gst-inspect-1.0
```

and now for some pipelines....

```commandline
gst-launch-1.0 filesrc location = /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! \
h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nvvideoconvert ! \
nvdspreprocess config-file= /opt/nvidia/deepstream/deepstream-6.0/sources/gst-plugins/gst-nvdspreprocess/config_preprocess.txt  ! \
nvinfer config-file-path= /opt/nvidia/deepstream/deepstream-6.0/samples/configs/deepstream-app/config_infer_primary.txt \
input-tensor-meta=1 batch-size=7  ! nvmultistreamtiler width=1920 height=1080 ! nvvideoconvert ! nvdsosd ! nveglglessink
```

```commandline
gst-launch-1.0 filesrc location = /opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! \
h264parse ! nvv4l2decoder ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! nvvideoconvert ! \
nvdspreprocess config-file= /opt/nvidia/deepstream/deepstream-6.0/sources/gst-plugins/gst-nvdspreprocess/config_preprocess.txt  ! \
nveglglessink
```

this one will replay the dataset to the screen
```bash
gst-launch-1.0 multifilesrc location="data/horizon/frame_%05d.png" caps="image/png,framerate=\(fraction\)2/1" ! pngdec ! videoconvert ! videorate ! autovideosink
```