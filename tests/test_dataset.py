import dataset
import numpy as np
from matplotlib import pyplot as plt
from time import sleep


def test_horizon_dataset_geometry():
    ds = dataset.HorizonDataSet(data_dir='data/test_geometry')
    x, y = ds[0]

    line1 = np.array([
        [0.0, 5.0],
        [0.0, 0.0]
    ])

    line2 = np.array([
        [0.0, 5.0],
        [0.0, 5.0]
    ])

    line3 = np.array([
        [0.0, 0.0],
        [0.0, 5.0]
    ])

    lines = np.stack([line1, line2, line3], axis=-1)

    expected = np.array([
        [1.0, 0.70710678, 0.0],
        [0.0, 0.70710678, 1.0]
    ]).T.copy().view(np.complex128)

    expected_angle = np.angle(expected.mean())

    assert np.allclose(lines, y['lines'])
    assert np.allclose(expected, y['complex'])
    assert np.allclose(expected_angle, y['angle'])
    plt.imshow(x.permute(1, 2, 0))
    plt.pause(1.0)


def test_horizon_dataset_visually():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    ds = dataset.HorizonDataSet(data_dir='data/test_visual', rotate=False)
    for img, label in ds:
        ax.clear()

        # plot the lines
        for i in range(label['lines'].shape[2]):
            ax.plot(label['lines'][0, :, i], label['lines'][1, :, i])

        # plot a cross at the center of screen
        center_h, center_w = img.shape[1] / 2, img.shape[2] / 2
        horizon = np.array([
            [-1., 1., 0.],
            [0, 0, 0.],
            [1., 1., 1.]
        ])

        angle = label['angle']
        r = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.],
            [0, 0, 1.]
        ])

        horizon = np.matmul(r, horizon)

        t = np.array([
            [100., 0, center_w],
            [0., 100., center_h],
            [0., 0., 1.]
        ])

        horizon = np.matmul(t, horizon)

        ax.plot(horizon[0], horizon[1], color='white', linewidth='4')

        ax.imshow(img.permute(1, 2, 0).byte())

        plt.pause(5.0)


def test_horizon_dataset_visually_with_rotations():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    ds = dataset.HorizonDataSet(data_dir='data/test_visual', rotate=True)
    for img, label in ds:
        ax.clear()

        # plot a cross at the center of screen
        center_h, center_w = img.shape[1] / 2, img.shape[2] / 2
        horizon = np.array([
            [-1., 1., 0.],
            [0., 0., 0.],
            [1., 1., 1.]
        ])

        angle = label['angle']
        r = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.],
            [0., 0., 1.]
        ])

        horizon = np.matmul(r, horizon)

        t = np.array([
            [100., 0, center_w],
            [0., 100., center_h],
            [0., 0., 1.]
        ])

        horizon = np.matmul(t, horizon)

        ax.plot(horizon[0], horizon[1], color='white', linewidth='4')

        ax.imshow(img.permute(1, 2, 0).byte())

        plt.pause(5.0)
