from setuptools import setup

setup(name='horizon',
      version='0.1',
      description='Horizon detection de',
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
            'lightning-flash[image]',
            'pytorch-lightning',
            'imageio'
      ],
      zip_safe=False)
