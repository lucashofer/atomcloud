import numpy as np


def find_square_mask(mask: np.ndarray) -> np.ndarray:
    """Find the square mask that completely encompasses the elliptical mask.

    Args:
        mask (np.ndarray): The elliptical mask.

        Returns:
            np.ndarray: The square mask.
    """

    # ChatGPT version
    rows, cols = np.where(mask)

    # Get the min and max row and column values
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Create a square mask that completely encompasses the elliptical mask
    square_mask = np.zeros_like(mask, dtype=bool)
    square_mask[min_row : max_row + 1, min_col : max_col + 1] = True
    return square_mask
