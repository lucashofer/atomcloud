# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:16:36 2022

@author: hofer
"""
import numpy as np

from atomcloud.analysis import rescale_1d_params
from atomcloud.common import registry
from atomcloud.functions.func_base import FunctionBase
from atomcloud.utils import fit_utils


# __all__ = ["FUNCTIONS1D"]

FUNCTIONS1D = registry.Registry("functions1d")

# TODO: there's an error in 1D integration when rescaling the parameters


class Function1DBase(FunctionBase):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def initial_seed(self, x, data):
        max_ind = np.argmax(data)
        a = data[max_ind]
        x0 = x[max_ind]
        std = (x[-1] - x[0]) * (1 / 12)
        return [a, std, x0]

    def rescale_parameters(self, params, scale):
        _, axis_scale, zscale = scale
        zinds = [0]
        xinds = [1, 2]
        return rescale_1d_params(params, zinds, xinds, axis_scale, zscale)

    def analyze_parameters(self, params: list[float]) -> dict:
        analysis_dict = {"int": self.integrate_function(params)}
        return analysis_dict

    def rescale_analysis_params(self, params: dict, scales: list) -> dict:
        num_scale, _, _ = scales
        params["int"] = params["int"] * num_scale
        return params

    def default_bounds(self):
        min_bounds = [0, 0, -np.inf]
        max_bounds = [np.inf, np.inf, np.inf]
        return [min_bounds, max_bounds]


class Gaussian1D(Function1DBase):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def create_gaussian1d(self, anp):
        def gaussian1d(x: np.ndarray, n0: float, sig: float, x0: float) -> np.ndarray:
            """1D Gaussian equation"""
            return n0 * anp.exp(-0.5 * ((x - x0) / sig) ** 2)

        return gaussian1d

    def create_function(self, anp):
        return self.create_gaussian1d(anp)

    def integrate_function(self, params):
        a, std, x0 = params
        return (2 * np.pi) ** 0.5 * a * std


class FixedEnhancedBose1D(Gaussian1D):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def create_polylog1d(self, anp, n_max=50):
        """This function needs help"""

        def polylog1d(z, gamma):
            """Poly log function in 1D."""
            Li = anp.zeros(z.shape)
            for n in range(1, n_max + 1):
                Li += z**n / n**gamma
            return Li

        return polylog1d

    def create_function(self, anp):
        polylog1d = self.create_polylog1d(anp)
        gaussian1d = self.create_gaussian1d(anp)

        def thermal_cloud(x, n0, sig, x0):
            """should probably check this t00"""
            gaussian = gaussian1d(x, 1, sig, x0)
            return n0 * polylog1d(gaussian, 5 / 2)

        return thermal_cloud

    def integrate_function(self, params):
        """should probably check this t00"""
        a, std, x0 = params
        poly_log = fit_utils.polylog_val(1, 3)
        return (2 * np.pi) ** 0.5 * a * std * poly_log


class EnhancedBose1D(FixedEnhancedBose1D):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def create_function(self, anp):
        polylog1d = self.create_polylog1d(anp)
        gaussian1d = self.create_gaussian1d(anp)

        def thermal_cloud(x, n0, sig, x0, fugacity):
            """should probably check this t00"""
            inside = fugacity * gaussian1d(x, 1, sig, x0)
            return n0 * polylog1d(inside, 5 / 2)

        return thermal_cloud

    def integrate_function(self, params):
        """should probably check this t00"""
        a, std, x0, fugacity = params
        poly_log = fit_utils.polylog_val(fugacity, 3)
        return (2 * np.pi) ** 0.5 * a * std * poly_log

    def default_bounds(self):
        min_bounds = [0, 0, -np.inf, 0]
        max_bounds = [np.inf, np.inf, np.inf, 1]
        return [min_bounds, max_bounds]


class Parabola1D(Function1DBase):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def create_parabola1d(self, anp):
        def parabola1d(x, n0, rx, x0):
            """1D parabola equation"""
            parabola = 1 - (x - x0) ** 2 / rx**2
            parabola = anp.where(parabola > 0, parabola, 0)
            return n0 * parabola

        return parabola1d

    def create_function(self, anp):
        return self.create_parabola1d(anp)

    def integrate_function(self, params):
        a, rx, x0 = params
        """1D parabola equation"""
        return (4 / 3) * a * rx


class ThomasFermi1D(Parabola1D):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def create_function(self, anp):
        parabola1d = self.create_parabola1d(anp)

        def thomas_fermi_bec(x, n0, rx, x0):
            """Integrated parabola"""
            parabola = parabola1d(x, 1, rx, x0)
            return n0 * parabola**2

        return thomas_fermi_bec

    def integrate_function(self, params):
        a, rx, x0 = params
        return (16 / 15) * a * rx


class FixedOffset1D(FunctionBase):
    """See FunctionBase for documentation"""

    def __init__(self):
        super().__init__()

    def create_function(self, anp):
        def fixed_offset(coords, foff):
            return foff * anp.ones(coords.shape)

        return fixed_offset

    def initial_seed(self, x, data):
        return [np.amin(data)]

    def rescale_parameters(self, params, scale):
        _, axis_scale, zscale = scale
        zinds = [0]
        xinds = []
        return rescale_1d_params(params, zinds, xinds, axis_scale, zscale)


FUNCTIONS1D.register("gaussian", Gaussian1D)
FUNCTIONS1D.register("parabola", Parabola1D)
FUNCTIONS1D.register("tf", ThomasFermi1D)
FUNCTIONS1D.register("ebose", EnhancedBose1D)
FUNCTIONS1D.register("febose", FixedEnhancedBose1D)
FUNCTIONS1D.register("foffset", FixedOffset1D)

# def get_default_seed(self, x, data):
#     """Calculates initial seed for a fit of a 1D cloud based only off the
#     1D coordinates and 1D data."""
#     dlength = len(data)
#     klength = dlength // 20
#     kernel = np.ones(klength)
#     cdata = np.convolve(data, kernel, 'same')
#     max_ind = np.argmax(cdata)
#     a = np.mean(data[max_ind-2:max_ind+2])
#     mu = x[max_ind]
#     std = (x[-1] - x[0]) * (1 / 10)
#     offset = np.amin(data)
#     func_seed = [a, std, mu, offset]
#     seed = [func_seed for i in range(self.num_funcs)]
#     return seed
