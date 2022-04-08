from matplotlib import pyplot as plt
from time import sleep
import imageio

fig, ax = plt.subplots()

im = imageio.imread('imageio:chelsea.png')
ax.imshow(im)
plt.pause(0.05)

for i in range(100):
    ax.clear()
    ax.set_xlim((0, 100))
    ax.barh('progress', i, height=10)
    fig.canvas.draw()
    sleep(0.5)