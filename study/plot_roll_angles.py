import numpy as np
import matplotlib.pyplot as plt
import dataset

N = 90
bottom = 8
max_height = 4

theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
radii = np.zeros_like(theta)

for x, labels in dataset.HorizonDataSet('../data/horizon'):
    d = theta - labels['angle']
    d = np.abs(d)
    cls = np.nanargmin(d)
    radii[cls] += 1

#radii = max_height*np.random.rand(N)
width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.show()