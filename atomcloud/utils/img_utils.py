# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 20:06:41 2022

@author: hofer
"""
import numpy as np


def get_coordinates(shape):
    """Returns meshgrid of x and y coordinates of shape

    Args:
        shape (tuple): shape of image

    Returns:
        tuple: meshgrid of x and y coordinates
    """

    height, width = shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    return np.meshgrid(x, y)


def aoi_coordinates(aoi):
    aoi = [int(round(i)) for i in aoi]
    xmin, ymin, xmax, ymax = aoi
    width = xmax - xmin
    height = ymax - ymin
    x = np.linspace(xmin, xmin + width - 1, width)
    y = np.linspace(ymin, ymin + height - 1, height)
    return np.meshgrid(x, y)


def img_aoi_data(img, aoi):
    aoi = [int(i) for i in aoi]
    XY_tuple = aoi_coordinates(aoi)
    img_data_cropped = crop_data(img, *aoi)
    return XY_tuple, img_data_cropped


def aoi_area(xmin, ymin, xmax, ymax):
    return (xmax - xmin) * (ymax - ymin)


def crop_data(data, xmin, ymin, xmax, ymax):
    data_cropped = data[ymin:ymax, xmin:xmax]
    return data_cropped


def crop_image_data(aoi, img_data, XY_tuple):
    X, Y = XY_tuple
    X_cropped = crop_data(X, *aoi)
    Y_cropped = crop_data(Y, *aoi)
    XY_cropped = (X_cropped, Y_cropped)
    img_data_cropped = crop_data(img_data, *aoi)
    return img_data_cropped, XY_cropped


def get_fit_data(img, aoi):
    XY_tuple = get_coordinates(img.shape)
    cimg, cXY = crop_image_data(aoi, img, XY_tuple)
    return cimg, cXY


def img_data_to_sums(XY_tuple, data, mask=None):
    x, y = tuple_coords1d(XY_tuple)
    if mask is not None:
        x, y, xsum, ysum = sum_data_mask(x, y, data, mask)
    else:
        xsum, ysum = sum_img_data(data, mask)
    return x, y, xsum, ysum


def tuple_coords1d(XY_tuple):
    X, Y = XY_tuple
    x = X[0]
    y = Y[:, 0]
    return x, y


def sum_img_data(data, mask=None):
    xsum = np.sum(data, axis=0)
    ysum = np.sum(data, axis=1)
    return xsum, ysum


def sum_data_mask(x, y, data, mask=None):
    if mask is not None:
        data = np.where(mask, data, mask)

    xsum = np.nansum(data, axis=0)
    ysum = np.nansum(data, axis=1)

    if mask is not None:
        xrows = ~np.all(np.isnan(data), axis=0)
        yrows = ~np.all(np.isnan(data), axis=1)
        xsum = xsum[xrows]
        ysum = ysum[yrows]
        x = x[xrows]
        y = y[yrows]

    return x, y, xsum, ysum
