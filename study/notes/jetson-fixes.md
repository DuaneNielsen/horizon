python bindings for tensorRT

```bash
export PYTHONPATH=/usr/lib/python3.6/dist-packages/tensorrt/:$PYTHONPATH
```

[pytorch for jetpack 4.6.1](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)

```bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt install libopenblas-base libopenmpi-dev libomp-dev
sudo pip3 install Cython
sudo pip3 install numpy ./torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

torchvision from source

```bash
cd ~
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.3 https://github.com/pytorch/vision torchvision
cd torchvision/
export BUILD_VERSION=0.11.3
sudo python3 setup.py install
```

