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