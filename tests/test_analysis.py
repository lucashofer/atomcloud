import numpy as np

from atomcloud.analysis.analysis_utils import cloud_temperature
from atomcloud.analysis.fit_metrics import calc_chi_squared
from atomcloud.analysis.image_scales import (
    all_axis_scales,
    convert_atom_number,
    img_axis_scales,
    od_nd_scale,
    optical_cross_section,
    pixel_scale,
)
from atomcloud.analysis.rescale_params import (
    rescale_1d_params,
    rescale_2d_params,
    rescale_mixed_axis,
    rescale_parameters,
)


# These are the tests for the analysis subpackage
# Mostly generated using Copilot and ChatGPT 3.5


def test_calc_chi_squared():
    num_parameters = 2
    actual_data = np.array([1, 2, 3, 4, 5])
    fit_data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    expected_chi_squared = np.sum((actual_data - fit_data) ** 2 / sigma**2)
    expected_chi_square_reduced = expected_chi_squared / (
        len(actual_data) - num_parameters
    )
    actual_chi_squared, actual_chi_square_reduced = calc_chi_squared(
        num_parameters, actual_data, fit_data, sigma
    )
    assert np.isclose(actual_chi_squared, expected_chi_squared)
    assert np.isclose(actual_chi_square_reduced, expected_chi_square_reduced)


def test_cloud_temperature():
    sigma = 1e-3
    mass = 166 * 1.660539e-27
    tof = 1e-3
    trap_freq = 1e6
    expected_temp = (sigma**2 * mass * trap_freq**2) / (
        1.38064852e-23 * (1 + (trap_freq * tof) ** 2)
    )
    actual_temp = cloud_temperature(sigma, mass, tof, trap_freq)
    assert abs(actual_temp - expected_temp) < 1e-6

    # Test with no trap frequency specified
    expected_temp = (sigma**2 * mass) / (1.38064852e-23 * tof**2)
    actual_temp = cloud_temperature(sigma, mass, tof)
    assert abs(actual_temp - expected_temp) < 1e-6


def test_rescale_parameters():
    params = [1, 2, 3, 4, 5]
    indices = [0, 3]
    scale = 2
    expected_params = [2, 2, 3, 8, 5]
    actual_params = rescale_parameters(params, indices, scale)
    assert actual_params == expected_params


def test_rescale_1d_params():
    params = [1, 2, 3, 4, 5]
    z_indices = [0, 3]
    x_indices = [1, 4]
    xscale = 2
    zscale = 3
    expected_params = [3, 4, 3, 12, 10]
    actual_params = rescale_1d_params(params, z_indices, x_indices, xscale, zscale)
    assert actual_params == expected_params


def test_rescale_mixed_axis():
    params = [1, 2, 3, 4, 5, 6]
    mixed_inds = [(1, 2), (4, 5)]
    xscale = 2
    yscale = 2
    expected_params = [1, 4, 6, 4, 10, 12]
    actual_params = rescale_mixed_axis(params, mixed_inds, xscale, yscale)
    assert actual_params == expected_params


def test_rescale_2d_params():
    params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    indices = [[0, 5], [1, 6], [2], [(3, 4), (7, 8)]]
    scales = [1, 2, 2, 3]
    theta_indices = [6, 7]
    expected_params = [2, 4, 9, 8, 10, 12, 14, 16, 18, 10]
    actual_params = rescale_2d_params(params, indices, scales, theta_indices)
    assert actual_params == expected_params


def test_convert_atom_number():
    atom_number = 100
    xscale = 1.2
    yscale = 2.3
    zscale = 3.4
    expected_atom_number = atom_number * xscale * yscale * zscale
    actual_atom_number = convert_atom_number(atom_number, xscale, yscale, zscale)
    assert np.isclose(actual_atom_number, expected_atom_number)


def test_optical_cross_section():
    lambd = 1e-6
    expected_cross_section = (3 * lambd**2) / (2 * np.pi)
    actual_cross_section = optical_cross_section(lambd)
    assert np.isclose(actual_cross_section, expected_cross_section)


def test_pixel_scale():
    pixel_length = 1e-6
    magnification = 2
    expected_scale = pixel_length / magnification
    actual_scale = pixel_scale(pixel_length, magnification)
    assert actual_scale == expected_scale


def test_od_nd_scale():
    optical_cross_section = 1e-12
    expected_nd = 1 / optical_cross_section
    actual_nd = od_nd_scale(optical_cross_section)
    assert actual_nd == expected_nd


def test_img_axis_scales():
    xpixel_length = 1e-6
    ypixel_length = 2e-6
    magnification = 2
    expected_scales = (xpixel_length / magnification, ypixel_length / magnification)
    actual_scales = img_axis_scales(xpixel_length, ypixel_length, magnification)
    assert actual_scales == expected_scales


def test_all_axis_scales():
    lambd = 1e-6
    xpixel_length = 1e-6
    ypixel_length = 2e-6
    magnification = 2
    expected_scale_dict = {
        "xscale": xpixel_length / magnification,
        "yscale": ypixel_length / magnification,
        "zscale": 1 / ((3 * lambd**2) / (2 * np.pi)),
    }
    expected_scale_list = [
        xpixel_length / magnification,
        ypixel_length / magnification,
        1 / ((3 * lambd**2) / (2 * np.pi)),
    ]
    actual_scale_dict, actual_scale_list = all_axis_scales(
        lambd, xpixel_length, ypixel_length, magnification
    )
    assert actual_scale_dict == expected_scale_dict
    assert actual_scale_list == expected_scale_list
