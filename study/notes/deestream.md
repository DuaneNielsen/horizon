install deepstream from [these instructions](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

verify by running the below

```
deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

you will need the rtp server

```
sudo apt-get install libgstrtspserver-1.0
```
