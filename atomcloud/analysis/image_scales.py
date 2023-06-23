# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 20:05:17 2022

@author: hofer
"""
import numpy as np


# __all__ = ["convert_atom_number", "all_axis_scales"]


def convert_atom_number(atom_number, xscale: float, yscale: float, zscale: float):
    """Rescale integrated values

    Args:
        atom_number: number of atoms
        xscale: scale factor for x axis
        yscale: scale factor for y axis
        zscale: scale factor for z axis

    Returns:
        rescaled atom number
    """
    return atom_number * xscale * yscale * zscale


def optical_cross_section(lambd: float) -> float:
    """
    Calculate the optical cross section of a laser beam on resonance.

    Args:
        lambd: wavelength of the laser beam in meters

    Returns:
        optical cross section in m^2
    """
    return (3 * lambd**2) / (2 * np.pi)


def pixel_scale(pixel_length: float, magnification: float = 1.0) -> float:
    """
    Calculates the image scaling value if we're only moving from
    pixels to meters. Includes the magnification factor of the imaging
    setup.

    Args:
        pixel_length: length of a pixel in meters
        magnification: magnification of the imaging setup

    Returns:
        scale factor for the image on the pixel axis
    """

    return pixel_length / magnification


def od_nd_scale(optical_cross_section: float):
    """
    Calculates the conversion from optical density to atom number
    density.

    Args:
        optical_cross_section: optical cross section of the laser beam

    Returns:
        atom number density
    """
    return 1 / optical_cross_section


def img_axis_scales(
    xpixel_length: float, ypixel_length: float, magnification: float = 1
) -> tuple[float, float]:
    """
    Calculates the scaling values along the image axes.

    Args:
        xpixel_length: x axis length of a pixel in meters
        ypixel_length: y axis length of a pixel in meters
        magnification: magnification of the imaging setup

    Returns:
        scale factor for the image on the pixel axes
    """
    xscale = pixel_scale(xpixel_length, magnification)
    yscale = pixel_scale(ypixel_length, magnification)
    return xscale, yscale


def all_axis_scales(
    lambd: float, xpixel_length: float, ypixel_length: float, magnification: float = 1
) -> tuple[dict[str, float], list[str]]:
    """
    Calculates the scaling values along all three axes.

    Args:
        lambd: wavelength of the laser beam in meters
        xpixel_length: x axis length of a pixel in meters
        ypixel_length: y axis length of a pixel in meters
        magnification: magnification of the imaging setup

    Returns:
        scale factor for the image on the pixel axes
    """
    xscale = pixel_scale(xpixel_length, magnification)
    yscale = pixel_scale(ypixel_length, magnification)

    cross_section = optical_cross_section(lambd)
    zscale = od_nd_scale(cross_section)
    scale_dict = {"xscale": xscale, "yscale": yscale, "zscale": zscale}
    return scale_dict, [xscale, yscale, zscale]
