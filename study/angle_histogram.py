import datamodules
from matplotlib import pyplot as plt

angles = []
for x, label in datamodules.HorizonDataSet('../data/horizon'):
    angles.append(label['angle'])

plt.hist(angles)
plt.show()