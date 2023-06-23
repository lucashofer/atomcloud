from typing import Iterable

import numpy as np


# __all__ = [
#     "check_1d_array",
#     "check_2d_array",
#     "check_2d_coords",
# ]


def check_1d_array(array):
    if isinstance(array, np.ndarray):
        if array.ndim != 1:
            raise ValueError("1D Fit coordinates should be 1D")
    else:
        raise TypeError("1D Fit coordinates should be a numpy array")


def check_2d_array(array):
    if isinstance(array, np.ndarray):
        if array.ndim != 2:
            raise ValueError("2D Fit coordinates should be 2D")
    else:
        raise TypeError("2D Fit coordinates should be a numpy array")


def check_2d_coords(coords):
    if isinstance(coords, Iterable):
        for coord in coords:
            check_2d_array(coord)
        for coord in coords[1:]:
            if coord.shape != coords[0].shape:
                raise ValueError(
                    "2D Fit coordinates should be the same \
                                shape"
                )
    else:
        raise TypeError(
            "2D Fit coordinates should be a \
                        list/tuple of numpy arrays"
        )
