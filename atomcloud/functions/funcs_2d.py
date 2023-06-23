# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:16:36 2022

@author: hofer
"""
from typing import Iterable

import numpy as np

from atomcloud.analysis import rescale_2d_params
from atomcloud.common import registry
from atomcloud.functions.func_base import FunctionBase
from atomcloud.functions.jax_funcs.spence import create_polylog2d
from atomcloud.utils import fit_utils, uncertain_utils as ucalcs


# __all__ = ["Function2DBase", "FUNCTIONS2D"]


FUNCTIONS2D = registry.Registry("functions2d")


def general_lab_widths(px, py, theta, key="sig"):
    """Given two orthogonal parameters which are rotated by theta with
    respect to the x and y axes, this function returns the widths in the
    lab frame, but in a dictionary format."""
    wxx, wyy = ucalcs.get_lab_widths(px, py, theta)
    return {key + "xx": wxx, key + "yy": wyy}


def general_analysis_rescale(analysis_params, scale, key=None):
    """General rescale for clouds which have only a single lab withds parameter
    and an integrated density."""
    int_key = "int"
    pxx_key = key + "xx"
    pyy_key = key + "yy"
    num_scale, xscale, yscale, zscale = scale
    analysis_params[pxx_key] = analysis_params[pxx_key] * xscale
    analysis_params[pyy_key] = analysis_params[pyy_key] * yscale
    analysis_params[int_key] = analysis_params[int_key] * num_scale
    return analysis_params


def general_rescale(params, scales):
    zinds = [0]
    xinds = [1]
    yinds = [2]
    mixed_inds = [[3, 4]]
    inds = [xinds, yinds, zinds, mixed_inds]
    th_ind = 5
    return rescale_2d_params(params, inds, scales, th_ind)


class Function2DBase(FunctionBase):
    """Inherits from FunctionBase and then adds the 2D specific
    function to create the coordinate transformation functions compatible
    with JAX.

    See FunctionBase for full documentation"""

    def __init__(self):
        """See FunctionBase for documentation"""
        super().__init__()

    def create_coord_funcs(self, anp):
        """Creates the coordinate transformation functions which are
        compatible with JAX due to the use of the creation wrapper function.
        Again, we can't use self in the function because it is not jittable

        Args:
            anp: The numpy or jax numpy module to use for the functions
             which are created.

        Returns:
            The overall coordinate transformation function for 2D coordinates
        """

        def rotate_coordinates2D(
            coords: Iterable[np.ndarray], theta: float
        ) -> Iterable[np.ndarray]:
            """Rotates the coordinates by theta

            Args:
                coordinates: The 2D coordinates to rotate
                theta: The angle to rotate by

            Returns:
                The rotated 2D coordinates
            """
            X, Y = coords
            shape = X.shape
            coords = anp.stack([anp.ravel(X), anp.ravel(Y)])
            R = anp.array(
                [[anp.cos(theta), -anp.sin(theta)], [anp.sin(theta), anp.cos(theta)]]
            )
            Xr, Yr = R.dot(coords)
            return anp.reshape(Xr, shape), anp.reshape(Yr, shape)

        def translate_coordinates2D(
            coords: Iterable[np.ndarray], x0: float, y0: float
        ) -> Iterable[np.ndarray]:
            """Translates the coordinates by x0 and y0

            Args:
                coordinates: The 2D coordinates to translate
                x0: The x translation
                y0: The y translation

            Returns:
                The translated 2D coordinates
            """
            X, Y = coords
            X = X - x0
            Y = Y - y0
            return (X, Y)

        def coordinate_transformation2D(
            coords: Iterable[np.ndarray], x0: float = 0, y0: float = 0, theta: float = 0
        ) -> Iterable[np.ndarray]:
            """Applies the translation and rotation to the coordinates

            Args:
                coordinates: The 2D coordinates to translate
                x0: The x translation
                y0: The y translation
                theta: The angle to rotate by

            Returns:
                The translated and rotated 2D coordinates
            """
            coords = translate_coordinates2D(coords, x0, y0)
            coords = rotate_coordinates2D(coords, theta)
            return coords

        return coordinate_transformation2D


class Gaussian2D(Function2DBase):
    """See FunctionBase for full documentation."""

    def __init__(self):
        """Initialize the function object along with the lab width key. See
        FunctionBase for full documentation."""
        super().__init__()
        self.sig_key = "sig"

    def create_gaussian2d(self, anp):
        """Creates the 2D gaussian function. This will be also used in
        the inherited ebose classes so it's a separate function."""
        coordinate_transformation2D = self.create_coord_funcs(anp)

        def gaussian2d(XY_tuple, n0, x0, y0, sigx, sigy, theta):
            X, Y = coordinate_transformation2D(XY_tuple, x0, y0, theta)
            inside = X**2 / sigx**2 + Y**2 / sigy**2
            gaussian_density = -0.5 * inside
            gaussian_density = n0 * anp.exp(gaussian_density)
            return gaussian_density

        return gaussian2d

    def create_function(self, anp):
        """Creates the 2D gaussian fitting function"""
        return self.create_gaussian2d(anp)

    def integrate_function(self, params):
        """Integrates the 2D gaussian function analytically"""
        n0, x0, y0, sigma_x, sigma_y, theta = params
        return 2 * np.pi * n0 * sigma_x * sigma_y

    def analyze_parameters(self, params):
        """Calculates the lab widths and the integrated density from the
        fitted parameters"""
        n0, x0, y0, sigma_x, sigma_y, theta = params
        analysis_dict = general_lab_widths(sigma_x, sigma_y, theta, key=self.sig_key)
        analysis_dict["int"] = self.integrate_function(params)
        return analysis_dict

    def rescale_parameters(self, params, scales):
        """Rescales the fitted parameters to the lab frame"""
        return general_rescale(params, scales)

    def rescale_analysis_params(self, analysis_params, scale):
        """Rescales the analysis parameters from the func above"""
        return general_analysis_rescale(analysis_params, scale, self.sig_key)

    def default_bounds(self):
        """Returns the default bounds for the 2D Gaussian function"""
        min_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
        max_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        return [min_bounds, max_bounds]


def integrate_ebose(n0, sigma_x, sigma_y, fugacity):
    """Integrates the ebose function analytically. However, currently
    the polylog value in the utils is a pretty good approximation. Need
    to make a general python polylog function that's JAX compatible in the
    future."""
    poly_log = fit_utils.polylog_val(fugacity, 3)
    return 2 * np.pi * n0 * sigma_x * sigma_y * poly_log


class EnhancedBose2D(Gaussian2D):
    """Enhanced Bose 2D cloud function object. This includes the fugacity in
    the fitting parameters. See FunctionBase for full documentation."""

    def __init__(self):
        """see CloudFunctionBase for full documentation"""
        super().__init__()

    def create_function(self, anp):
        """Creates the 2D enhanced Bose fitting function. The polylog function
        which is just spence(1-z) for the 2D case is used
        (see jaxfuncs.spence.py). Also utilizes the gaussian2d function from
        the Gaussian2D class which is inherited."""
        polylog2d = create_polylog2d(anp)
        coordinate_transformation2D = self.create_coord_funcs(anp)
        gaussian2d = self.create_gaussian2d(anp)

        def thermal_cloud(XY_tuple, n0, x0, y0, sigx, sigy, theta, fug):
            XY_tuple = coordinate_transformation2D(XY_tuple, x0, y0, theta)
            gaussian_density = gaussian2d(XY_tuple, 1, 0, 0, sigx, sigy, 0)
            return n0 * polylog2d(fug * gaussian_density)

        return thermal_cloud

    def analyze_parameters(self, params):
        """Analysis includes the integrated density and the sigma labwidths."""
        n0, x0, y0, sigma_x, sigma_y, theta, fug = params
        analysis_dict = general_lab_widths(sigma_x, sigma_y, theta, key=self.sig_key)
        analysis_dict["int"] = integrate_ebose(n0, sigma_x, sigma_y, fug)
        return analysis_dict

    def default_bounds(self):
        """Default bounds for the enhanced Bose function are the same as for
        the Gaussian except the fugacity is bounded between 0 and 1."""
        min_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf, 0]
        max_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]
        return [min_bounds, max_bounds]


class FixedEnhancedBose2D(Gaussian2D):
    """Fixed enhanced Bose 2D cloud function object. We assume that for this
    fit function we are near quantum degeneracy and the fugacity is 1.
    See CloudFunctionBase for full documentation"""

    def __init__(self):
        """Initialize the function object with the numpy or jax.numpy type"""
        super().__init__()

    def create_function(self, anp):
        """Creates the 2D fixed enhanced Bose fitting function see the
        non-fixed version for more details of the polylog function and
        gaussian2d function."""
        polylog2d = create_polylog2d(anp)
        gaussian2d = self.create_gaussian2d(anp)
        coordinate_transformation2D = self.create_coord_funcs(anp)

        def thermal_cloud(XY_tuple, n0, x0, y0, sigx, sigy, theta):
            XY_tuple = coordinate_transformation2D(XY_tuple, x0, y0, theta)
            gaussian_density = gaussian2d(XY_tuple, 1, 0, 0, sigx, sigy, 0)
            return n0 * polylog2d(gaussian_density)

        return thermal_cloud

    def analyze_parameters(self, params):
        """Analysis includes the integrated density and the sigma labwidths."""
        n0, x0, y0, sigma_x, sigma_y, theta = params
        analysis_dict = general_lab_widths(sigma_x, sigma_y, theta, key=self.sig_key)
        analysis_dict["int"] = integrate_ebose(n0, sigma_x, sigma_y, 1)
        return analysis_dict


class Parabola2D(Function2DBase):
    """Parabola 2D cloud function object. This is a simple parabola function
    and not for the integrated case, although it can be used as an
    approximation"""

    def __init__(self):
        """Initialize the function object with the numpy or jax.numpy type.
        Additionally, we define the key for the labwidths as 'r' for the
        radius of the upside down parabola or for the (inherited)
        thomas-fermi radius."""
        super().__init__()
        self.sig_key = "r"

    def create_parabola2d(self, anp):
        """Creates the 2D parabola function. This is upside down and
        does not go lower than 0. Using the np.where or jnp.where which makes
        it JAX compatible. The parabola is also used for the Thomas-Fermi
        object which is why it is a separate function."""
        coordinate_transformation2D = self.create_coord_funcs(anp)

        def parabola2d(XY_tuple, n0, x0, y0, rx, ry, theta):
            X, Y = coordinate_transformation2D(XY_tuple, x0, y0, theta)
            parabola = 1 - X**2 / rx**2 - Y**2 / ry**2
            parabola = anp.where(parabola > 0, parabola, 0)
            parabola = n0 * parabola
            return parabola

        return parabola2d

    def create_function(self, anp):
        """Creates the 2D parabola fitting function."""
        return self.create_parabola2d(anp)

    def integrate_function(self, params):
        """Integrates the 2D parabola function. This is done analytically"""
        n0, x0, y0, Rx, Ry, theta = params
        return (1 / 2) * np.pi * n0 * Rx * Ry

    def analyze_parameters(self, params):
        """Analysis includes the integrated density and the radius
        labwidths."""
        n0, x0, y0, Rx, Ry, theta = params
        analysis_dict = general_lab_widths(Rx, Ry, theta, key=self.sig_key)
        analysis_dict["int"] = self.integrate_function(params)
        return analysis_dict

    def rescale_parameters(self, params, scales):
        """Rescales the parameters for the 2D parabola function. This is
        the same as the 2d gaussian function."""
        return general_rescale(params, scales)

    def rescale_analysis_params(self, analysis_params, scale):
        """Rescales the analysis parameters for the 2D parabola function.
        This is the same as the 2d gaussian function."""
        return general_analysis_rescale(analysis_params, scale, self.sig_key)

    def default_bounds(self):
        """Default bounds for the 2D parabola function are the same as for
        the Gaussian."""
        min_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
        max_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        return [min_bounds, max_bounds]


class ThomasFermi2D(Parabola2D):
    """Thomas-Fermi 2D cloud function object. This is a simple parabola
    but with a different exponent (3/2) since in absorption imaging we are
    integrating along the imaging beam axis."""

    def __init__(self):
        """Initialize the function object with the numpy or jax.numpy type."""
        super().__init__()

    def create_function(self, anp):
        """Creates the 2D Thomas-Fermi fitting function."""
        parabola2d = self.create_parabola2d(anp)
        coordinate_transformation2D = self.create_coord_funcs(anp)

        def thomas_fermi_bec(XY_tuple, n0, x0, y0, rx, ry, theta):
            XY_tuple = coordinate_transformation2D(XY_tuple, x0, y0, theta)
            parabola = parabola2d(XY_tuple, 1, 0, 0, rx, ry, 0)
            bec_density = n0 * parabola ** (3 / 2)
            return bec_density

        return thomas_fermi_bec

    def integrate_function(self, params):
        """Integrates the 2D Thomas-Fermi function. This is done analytically"""
        n0, x0, y0, Rx, Ry, theta = params
        return (2 / 5) * np.pi * n0 * Rx * Ry


class FixedOffset2D(FunctionBase):
    """Fixed offset 2D cloud function object. This is a simple offset function
    which can be combined with those functions objects above."""

    def __init__(self):
        """Initialize the function object with the numpy or jax.numpy type."""
        super().__init__()

    def create_function(self, anp):
        """Creates the 2D fixed offset fitting function."""

        def fixed_offset(coords, foff):
            return foff * anp.ones(coords[0].shape)

        return fixed_offset

    def integrate_function(self, params):
        """There is no analytic integration"""
        return np.nan

    def rescale_parameters(self, params, scale):
        """We only need to rescale the offset."""
        _, _, _, zscale = scale
        params[0] = params[0] * zscale
        return params

    def default_bounds(self):
        """This can be absolutely anything between -inf and inf."""
        return [[-np.inf], [np.inf]]


# The 2d function objects are stored in a dictionary which is then used
# elsewhere to create the multifunction objects.

FUNCTIONS2D.register("gaussian", Gaussian2D)
FUNCTIONS2D.register("parabola", Parabola2D)
FUNCTIONS2D.register("tf", ThomasFermi2D)
FUNCTIONS2D.register("ebose", EnhancedBose2D)
FUNCTIONS2D.register("febose", FixedEnhancedBose2D)
FUNCTIONS2D.register("foffset", FixedOffset2D)
