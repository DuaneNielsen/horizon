install deepstream from [these instructions](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

verify by running the below

```
deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

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
