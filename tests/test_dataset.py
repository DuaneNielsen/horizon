import dataset
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import torch


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


def r_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.],
        [0, 0, 1.]
    ])


def test_horizon_dataset_visually():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    ds = dataset.HorizonDataSet(data_dir='data/test_visual', rotate=False, num_classes=16, image_size=64)
    for img, label in ds:
        ax.clear()

        # plot the lines
        for i in range(label['lines'].shape[2]):
            ax.plot(label['lines'][0, :, i], label['lines'][1, :, i], linewidth=2)

        # plot a cross at the center of screen
        center_h, center_w = img.shape[1] / 2, img.shape[2] / 2
        x_ruler = np.array([
            [-1., 1., 0.],
            [0, 0, 0.],
            [1., 1., 1.]
        ])

        half_x_ruler = np.array([
            [0., 1., 0.],
            [0, 0, 0.],
            [1., 1., 1.]
        ])


        angle = label['angle']
        r_angle = r_matrix(angle)
        r_discrete_min = r_matrix(label['discrete_min'])
        r_discrete_max = r_matrix(label['discrete_max'])

        horizon = np.matmul(r_angle, x_ruler)
        discrete_min = np.matmul(r_discrete_min, half_x_ruler)
        discrete_max = np.matmul(r_discrete_max, half_x_ruler)

        t = np.array([
            [100., 0, center_w],
            [0., 100., center_h],
            [0., 0., 1.]
        ])

        horizon = np.matmul(t, horizon)
        discrete_min = np.matmul(t, discrete_min)
        discrete_max = np.matmul(t, discrete_max)

        ax.plot(horizon[0], horizon[1], color='white', linewidth='4')
        ax.plot(discrete_min[0], discrete_min[1], color='blue', linewidth='2')
        ax.plot(discrete_max[0], discrete_max[1], color='red', linewidth='2')

        for c in range(3):
            img[c] = img[c] - img[c].min()
            img[c] = img[c] * 256 / img[c].max()

        ax.imshow(img.permute(1, 2, 0).byte())

        plt.pause(5.0)


def test_horizon_dataset_visually_with_rotations():
    fig, ax = plt.subplots()
    fig.canvas.draw()
    ds = dataset.HorizonDataSet(data_dir='data/test_visual', rotate=True, image_size=64)
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

        #complex_mean = torch.complex(torch.sin(angle), torch.cos(angle))
        complex_mean = label['complex_mean']
        ax.plot([center_w, complex_mean.real * center_w + center_w],
                [center_h, complex_mean.imag * center_h + center_h],
                color='green', linewidth='2')

        for c in range(3):
            img[c] = img[c] - img[c].min()
            img[c] = img[c] * 256 / img[c].max()

        ax.imshow(img.permute(1, 2, 0).byte())

        plt.pause(5.0)
