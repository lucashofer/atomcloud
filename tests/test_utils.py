import numpy as np

from atomcloud.utils.mask_utils import find_square_mask


def test_find_square_mask():
    # create an elliptical mask with a major axis of 10 pixels and a minor axis
    # of 5 pixels
    mask = np.zeros((20, 20), dtype=bool)
    y, x = np.ogrid[-10:10, -10:10]
    mask[(x / 5) ** 2 + (y / 10) ** 2 <= 1] = True

    # get the expected square mask
    expected_square_mask = np.zeros((20, 20), dtype=bool)
    expected_square_mask[0 : 20 + 1, 5 : 15 + 1] = True

    # check that the actual and expected square masks are equal
    actual_square_mask = find_square_mask(mask)
    np.testing.assert_array_equal(actual_square_mask, expected_square_mask)
