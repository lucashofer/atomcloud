from typing import Optional

import numpy as np


# __all__ = ["calc_chi_squared"]


def calc_chi_squared(
    num_parameters: int,
    actual_data: np.ndarray,
    fit_data: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> list[float]:
    """
    Calculates the chi squared value and the reduced chi squared value for a
    fit.

    Args:
        num_parameters: The number of parameters in the fit.
        actual_data: The actual data that was fit.
        fit_data: Fit corresponding to original coord points based on fit
            parameters.
        sigma: The standard deviation of the data can be used to weight the
            chi squared value.

    Returns:
        The chi squared value and the reduced chi squared value.

    """
    if sigma is None:
        sigma = np.ones(len(actual_data))
    num_observations = len(actual_data)
    nu = num_observations - num_parameters
    chi_img = (actual_data - fit_data) ** 2 / sigma**2
    chi_squared = np.sum(chi_img)
    chi_square_reduced = chi_squared / nu
    return chi_squared, chi_square_reduced


# def calculate_sigma(sim_img, aoi_img, offset):
#   sigma_shotnoise = ((sim_img - offset))**.5 / 16 **.5
#   # sigma_shotnoise = (aoi_img - offset)**.5
#   # print(sigma_shotnoise)

#   # sigma_shotnoise = (sim_img)**.5

#   sigma_quantization = 1 / 12**.5  # https://www.sciencedirect.com/science/article/pii/B978012374457900007X
#   sigma = (sigma_shotnoise**2 + sigma_quantization**2)**.5
#   sigma = (sigma_shotnoise**2 + sigma_background**2)**.5
#   # sigma_background
#   # sigma = (sigma_shotnoise**2)**.5
#   # sigma = sigma_quantization
#   return sigma
