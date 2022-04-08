import datamodules
import numpy as np


def test_horizon_dataset():
    ds = datamodules.HorizonDataSet(data_dir='data/test_geometry')
    x = ds[0]

    expected = np.array([
        [1.0, 0.70710678, 0.0],
        [0.0, 0.70710678, 1.0]
    ])

    assert np.allclose(expected, x)