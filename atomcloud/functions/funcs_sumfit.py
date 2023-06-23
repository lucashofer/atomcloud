""" A registry of all the sumfunctions used to convert between 1D and 2D
    functions. If a user has a custom function they must define both the 1D
    and 2D versions of the function as well as this sumfunction object. The
    user should inherit from the SumFunctionBase class and define the
    sumfunction and the inverse sumfunction. The user should then add the
    sumfunction to the SUMFUNCTIONS registry with a key. This function can
    then be called using the sumfit object.

    from atomcloud.functions import SumFunctionBase, SUMFUNCTIONS

    class Lorentz(SumFitBaseFunc):

        def convert_2d_sum(self, XY_tuple, params):
            amp2d, x0, y0, sigmax, sigmay, theta = params
            amps1d = amps_2D_to_1D(amp2d, sigmax, sigmay, self.gaussian1D_amp)
            xsum_amp, ysum_amp = amps1d
            return [xsum_amp, sigmax, x0], [ysum_amp, sigmay, y0]

        def convert_sum_2D(self, XY_tuple, xparams, yparams):
            ax, sigmax, x0 = xparams
            ay, sigmay, y0 = yparams
            amp = amps_1D_to_2D(ax, ay, sigmax, sigmay, self.gaussian1D_amp)
            return [amp, x0, y0, sigmax, sigmay, 0]

    SUMFUNCTIONS.register('lorentz', Lorentz)

    """

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from atomcloud.common import registry
from atomcloud.utils.fit_utils import calc_diff_elements


# __all__ = ["SumFitBaseFunc", "SUMFUNCTIONS"]

SUMFUNCTIONS = registry.Registry("sumfunctions")


class SumFitBaseFunc(ABC):
    @abstractmethod
    def convert_2d_sum(
        self, coords: Iterable[np.ndarray], params: list[float]
    ) -> tuple[list[float]]:
        """
        Converts 2D function parameters into two sets of 1D parameters along
        the x and y axes respectively.
        Args:
            coords: The 2D coordinates of the data being fit
            params: The 2D parameters of the function being fit

        Returns:
            The 1D function parameters along the x and y axes
        """
        return [], []

    @abstractmethod
    def convert_sum_2D(
        self, coords: Iterable[np.ndarray], xparams: list[float], yparams: list[float]
    ) -> tuple[list[float]]:
        """
        Takes the 1D fit parameters of the sums along the x and y axes and
        converts them into 2D parameters.
        Args:
            coords: The 2D coordinates of the data being fit
            xparams: The 1D sum fit parameters along the x axis
            yparams: The 1D sum fit parameters along the y axis

        Returns:
            The 2D function parameters
        """
        return []


def amps_2D_to_1D(
    amp: float, sigma_x: float, sigma_y: float, scalar: callable(float)
) -> tuple[float, float]:
    """Converts 2D amplitudes to 1D sum amplitudes. For the 1D equations
    used for fitting the atom cloud integrating along one axis means the
    2D peak amplitude get's multiplied by a scalar which is a function of
    the functions radius. These scalars have been calculated in mathematica
    and are found in the ConvertAmp object.

    Args:
        amp: The 2D amplitude
        sigma_x: The sigma along the x axis
        sigma_y: The sigma along the y axis
        scalar: A function scales between sum and 2D amplitudes

    Returns:
        The 1D amplitudes along the x and y axes
    """
    xsum_amp = amp * (1 / scalar(sigma_y))
    ysum_amp = amp * (1 / scalar(sigma_x))
    return xsum_amp, ysum_amp


def amps_1D_to_2D(
    x_amp: float,
    y_amp: float,
    sigma_x: float,
    sigma_y: float,
    scalar_function: callable(float),
) -> float:
    """Converts 1D amplitudes to 2D sum amplitudes and averages them
    to get a single converted offset (see amps_1D_to_2D function for more
    details on the conversion).

    Args:
        x_amp: The 1D sum amplitude along the x axis
        y_amp: The 1D sum amplitude along the y axis
        sigma_x: The sigma along the x axis
        sigma_y: The sigma along the y axis
        scalar_function: A function that scales between sum and 2D amplitudes

    Returns:
        The 2D amplitude

    """
    x_amp_2d = x_amp * scalar_function(sigma_y)
    y_amp_2d = y_amp * scalar_function(sigma_x)
    amplitude = (x_amp_2d + y_amp_2d) / 2
    return amplitude


class Gaussian(SumFitBaseFunc):
    """See SumFitBaseFunc for documentation"""

    def gaussian1D_amp(self, std):
        """1D Gaussian equation"""
        return 1 / (std * (2 * np.pi) ** 0.5)

    def convert_2d_sum(self, XY_tuple, params):
        amp2d, x0, y0, sigmax, sigmay, theta = params
        amps1d = amps_2D_to_1D(amp2d, sigmax, sigmay, self.gaussian1D_amp)
        xsum_amp, ysum_amp = amps1d
        return [xsum_amp, sigmax, x0], [ysum_amp, sigmay, y0]

    def convert_sum_2D(self, XY_tuple, xparams, yparams):
        ax, sigmax, x0 = xparams
        ay, sigmay, y0 = yparams
        amp = amps_1D_to_2D(ax, ay, sigmax, sigmay, self.gaussian1D_amp)
        return [amp, x0, y0, sigmax, sigmay, 0]


class Parabola(SumFitBaseFunc):
    """See SumFitBaseFunc for documentation"""

    def parabola1D_amp(self, rx):
        """1D parabola equation"""
        return (3 / 4) * (1 / rx)

    def convert_2d_sum(self, XY_tuple, params):
        amp2d, x0, y0, Rx, Ry, theta = params
        amps1d = amps_2D_to_1D(amp2d, Rx, Ry, self.parabola1D_amp)
        xsum_amp, ysum_amp = amps1d
        return [xsum_amp, Rx, x0], [ysum_amp, Ry, y0]

    def convert_sum_2D(self, XY_tuple, xparams, yparams):
        ax, rx, x0 = xparams
        ay, ry, y0 = yparams
        amp = amps_1D_to_2D(ax, ay, rx, ry, self.parabola1D_amp)
        return [amp, x0, y0, rx, ry, 0]


class ThomasFermi(SumFitBaseFunc):
    """See SumFitBaseFunc for documentation"""

    def tf1d_amp(self, rx):
        """Integrated parabola equation this needs fixing too"""
        return 1 / ((16 / 15) * rx)

    def convert_2d_sum(self, XY_tuple, params):
        amp2d, x0, y0, Rx, Ry, theta = params
        amps1d = amps_2D_to_1D(amp2d, Rx, Ry, self.tf1d_amp)
        xsum_amp, ysum_amp = amps1d
        return [xsum_amp, Rx, x0], [ysum_amp, Ry, y0]

    def convert_sum_2D(self, XY_tuple, xparams, yparams):
        ax, rx, x0 = xparams
        ay, ry, y0 = yparams
        amp = amps_1D_to_2D(ax, ay, rx, ry, self.tf1d_amp)
        return [amp, x0, y0, rx, ry, 0]


class FixedOffset(SumFitBaseFunc):
    """See SumFitBaseFunc for documentation"""

    def convert_2d_sum(self, XY_tuple, params):
        foff = params[0]
        ylength, xlength = XY_tuple[0].shape
        dx, dy = calc_diff_elements(XY_tuple)
        # integrate along x and y axes
        xsum_offset = (ylength * foff) * dy
        ysum_offset = (xlength * foff) * dx
        return [xsum_offset], [ysum_offset]

    def convert_sum_2D(self, XY_tuple, xparams, yparams):
        """Reverse integration of constant offset by dividing by number of
        pixels on each integration axis. Then average offsets on each axis."""
        xfoff = xparams[0]
        yfoff = yparams[0]
        ylength, xlength = XY_tuple[0].shape
        dx, dy = calc_diff_elements(XY_tuple)

        xcut_offset = xfoff / (ylength * dy)
        ycut_offset = yfoff / (xlength * dx)
        offset = (xcut_offset + ycut_offset) / 2  # average
        return [offset]


# register all the sum fit conversions classes
SUMFUNCTIONS.register("gaussian", Gaussian)
SUMFUNCTIONS.register("parabola", Parabola)
SUMFUNCTIONS.register("tf", ThomasFermi)
SUMFUNCTIONS.register("foffset", FixedOffset)
