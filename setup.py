from setuptools import setup

setup(name='horizon',
      version='0.1',
      description='Horizon detection demo',
      url='http://github.com/duanenielsen/horizon',
      author='duanenielsen',
      author_email='duane.nielsen.rocks@gmail.com',
      license='MIT',
      packages=['horizon'],
      install_requires=[
            'matplotlib',
            'numpy',
            'torch',
            'torchvision',
            'pytorch-lightning',
            'torchmetrics',
            'imageio',
            'imageio_ffmpeg',
            'timm',
            'opencv-python',
            'pillow'
      ],
      zip_safe=False)
