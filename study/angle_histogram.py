import dataset
from matplotlib import pyplot as plt

angles = []
for x, label in dataset.HorizonDataSet('../data/horizon'):
    angles.append(label['angle'])

plt.hist(angles)
plt.show()