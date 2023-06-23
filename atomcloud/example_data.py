# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:08:19 2022

@author: hofer
"""


import numpy as np

from atomcloud.functions.multi_funcs import MultiFunction2D


def get_coordinates(width, height):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    return X, Y


def get_bimodal_distribution_parameters(XY_tuple):
    coord_length = len(XY_tuple[0])

    x0 = coord_length / 3
    y0 = coord_length / 2
    offset = 0.1
    theta = np.pi / 3

    wx = coord_length / 10
    sigma_x = wx
    sigma_y = 1.5 * sigma_x
    Rx = coord_length / 15
    Ry = 0.8 * Rx

    n_thermal = 0.5
    n_bec = 1.25

    gt_ex = [n_thermal, x0, y0, sigma_x, sigma_y, theta]
    ex_off = [offset]
    gt_bec = [n_bec, x0, y0, Rx, Ry, theta]

    return gt_ex, gt_bec, ex_off


def get_density(XY_tuple, thermal_params, bec_params, off_params):
    th = MultiFunction2D(["febose"])
    tf = MultiFunction2D(["tf"])
    off = MultiFunction2D(["foffset"])
    thermal_density = th.function(XY_tuple, [thermal_params])
    bec_density = tf.function(XY_tuple, [bec_params])
    offset = off.function(XY_tuple, [off_params])
    combined_density = thermal_density + bec_density + offset
    return combined_density, thermal_density, bec_density, offset


def get_example_data(coord_length=300):
    XY_tuple = get_coordinates(coord_length, coord_length)
    all_params = get_bimodal_distribution_parameters(XY_tuple)
    thermal_params, bec_params, off_params = all_params
    densities = get_density(XY_tuple, *all_params)
    return XY_tuple, densities, all_params
