import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot()

seq_id = 1

# load nvinfer input from file and verify its the same
infer_input = np.fromfile(
    f'gstnvdsinfer_uid-01_layer-modelInput_batch-{seq_id:010}_batchsize-01.bin',
    dtype=np.float32).reshape(3, 64, 64).transpose(1, 2, 0)

# load nvinfer output from file to verify results are the same as pytorch output
infer_output = np.fromfile(
    f'gstnvdsinfer_uid-01_layer-modelOutput_batch-{seq_id:010}_batchsize-01.bin',
    dtype=np.float32)

prediction = np.argmax(infer_output)
ax.text(3, 8, f'prediction: {prediction}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.9, 'pad': 10})

ax.imshow(infer_input)


plt.show()


