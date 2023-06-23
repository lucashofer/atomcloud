# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 17:29:59 2022

@author: hofer
"""
import numpy as np


def is_nested_list(obj, level):
    if level == 0:
        return isinstance(obj, list)
    elif isinstance(obj, list):
        return all(is_nested_list(item, level - 1) for item in obj)
    else:
        return False


def calc_diff_elements(coords):
    """Calculate the difference between each element in the x and y
    arrays. This is used to calculate the average pixel size along each
    axis."""
    X, Y = coords
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    return dx, dy


def rotate_coordinates2D(
    coordinates: tuple[np.ndarray, np.ndarray], theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotates coordinates by theta radians

    Args:
        coordinates: tuple of x and y coordinates
        theta: angle in radians

    Returns:
        tuple: rotated x and y coordinates
    """
    X, Y = coordinates
    shape = X.shape
    coords = np.stack([np.ndarray.flatten(X), np.ndarray.flatten(Y)])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Xr, Yr = R.dot(coords)
    return np.reshape(Xr, shape), np.reshape(Yr, shape)


def translate_coordinates2D(coords, x0, y0):
    """Translates coordinates by x0 and y0"""
    X, Y = coords
    X = X - x0
    Y = Y - y0
    return (X, Y)


def coordinate_transformation2D(XY_tuple, x0=0, y0=0, theta=0):
    """Transforms coordinates by first performing a translation and then a
    rotation"""
    XY_tuple = translate_coordinates2D(XY_tuple, x0, y0)
    XY_tuple = rotate_coordinates2D(XY_tuple, theta)
    return XY_tuple


def generate_elliptical_mask2D(XY_tuple, x0, y0, sig_x, sig_y, theta, scale=1):
    """Currently does mask edge at wx (2 * sigmax), but change it to sigmax"""
    X, Y = coordinate_transformation2D(XY_tuple, x0, y0, theta)
    p1 = X**2 / (scale * sig_x) ** 2
    p2 = Y**2 / (scale * sig_y) ** 2
    ecalc = p1 + p2  # basically ellipse calculation for every set of pixel coords
    mask = ecalc <= 1  # if the value is at or inside radius then set to one
    return mask


def approximate_jonquieress_function(z, gamma, n_max=10):
    Li = np.zeros(z.shape)
    for n in range(1, n_max + 1):
        Li += z**n / n**gamma
    return Li


def polylog_val(z, gamma, n_max=100):
    Li = 0
    zn = 1
    for n in range(1, n_max + 1):
        zn *= z
        Li += zn / n**gamma
    return Li


def get_array_bounds(data):
    dmin = np.amin(data)
    dmax = np.amax(data)
    ddelta = dmax - dmin
    dcenter = dmin + ddelta / 2
    return dmin, dmax, ddelta, dcenter


def get_masked_data(XY_tuple, data, mask):
    masked_data = data[mask]
    X, Y = XY_tuple
    masked_tuple = (X[mask], Y[mask])
    return masked_tuple, masked_data


def get_lab_widths(px, py, theta):
    """Given two orthogonal parameters which are rotated by theta with
    respect to the x and y axes, this function returns the widths in the
    lab frame."""
    pxx = ((px * np.cos(theta)) ** 2 + (py * np.sin(theta)) ** 2) ** 0.5
    pyy = ((px * np.sin(theta)) ** 2 + (py * np.cos(theta)) ** 2) ** 0.5
    return [pxx, pyy]


def get_wrapped_angle(angle, angle_range=np.pi, min_angle=-np.pi / 2):
    """wrap angles to be within a certain angle range"""
    max_angle = min_angle + angle_range
    return (angle + max_angle) % angle_range + min_angle


# def fix_theta(i0, x0, y0, sigmax, sigmay, theta, offset):
#     "this function works, but needs to be analyzed and condensed"
#     pi_val = np.pi
#     # find angle difference
#     mdiff = 0 - theta
#     int_multi1 = round(mdiff / pi_val)
#     if mdiff > 0:
#         int_multi2 = int_multi1 + 1
#     else:
#         int_multi2 = int_multi1 - 1
#     val1 = theta + int_multi1 * pi_val
#     val2 = theta + int_multi2 * pi_val
#     diff1 = abs(0 - val1)
#     diff2 = abs(0 - val2)
#     if diff1 < diff2:
#         theta += int_multi1 * pi_val
#     else:
#         theta += int_multi2 * pi_val
#     return [i0, x0, y0, sigmax, sigmay, theta, offset]


# def major_minor_fix(fit_params, fit_errors):
#     fit_params = copy.deepcopy(fit_params)
#     fit_errors = copy.deepcopy(fit_errors)
#     sigmax, sigmay, theta = fit_params[3:6]
#     sigmax_error, sigmay_error = fit_params[3:5]
#     if sigmax < sigmay:
#         sigmax, sigmay = sigmay, sigmax
#         sigmax_error, sigmay_error = sigmay_error, sigmax_error
#         theta += np.pi / 2

#     fit_params[3:6] = [sigmax, sigmay, theta]
#     fit_errors[3:5] = [sigmax_error, sigmay_error]

#     return fit_params, fit_errors
