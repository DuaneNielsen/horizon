import imageio as iio
from matplotlib import pyplot as plt

reader = iio.get_reader('/home/duane/Downloads/drone_fpv1.mp4')
for i, im in enumerate(reader):
    if i % 10 == 0:
        plt.clf()
        plt.imshow(im)
        plt.pause(0.0001)
