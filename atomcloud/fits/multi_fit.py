# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:49:49 2022

@author: hofer
"""
import time
from abc import ABC
from typing import Iterable, Optional, Union

import numpy as np
import uncertainties
from scipy.optimize import curve_fit

from atomcloud.analysis import calc_chi_squared
from atomcloud.utils import fit_utils


try:
    import jaxfit

    JAX = True
except ImportError:
    JAX = False

# __all__ = ["MultiFunctionFit"]


class MultiFunctionFit(ABC):
    def __init__(
        self,
        function_names: list[str],
        multi_func: object,
        func_registry: dict,
        fit_label: str,
        max_nfev_scalar: int = 50,
        constraints: Optional[list[str]] = None,
        scipy_length: int = 1e3,
        fixed_length: Optional[str] = None,
    ) -> None:
        """
        This class is used to create a multi-function fit object using either
        numpy or JAX. The resulting object is then used to do a multi-function
        fit to the data using either scipy or JAXFit.

        Args:
            function_names: names of the functions to be used in the fit
            dimensions: number of dimensions of the data
            constraints: list of constraints to be used in the multi-function
            scipy_length: length of data to use scipy over jax
            fixed_length: fixed length in JAXFit
        """

        self.function_names = function_names
        self.constraints = constraints
        self.scipy_length = scipy_length
        self.fixed_length = fixed_length
        self.func_registry = func_registry
        self.fit_label = fit_label

        # make rename func to func and make cloud_func obj _call__ function()
        self.func = multi_func(function_names, constraints, use_jax=False)
        self.fit_object_init(multi_func)

        if isinstance(max_nfev_scalar, int):
            self.max_nfev = max_nfev_scalar * self.func.num_args
        else:
            raise TypeError(
                f"max_nfev_scalar must be an integer, not {type(max_nfev_scalar)}"
            )

        self.info_keys = [
            "fit_type",
            "equations",
            "constraints",
            "params",
            "fit_metrics",
            "data_sum",
        ]

    def fit_object_init(self, multi_func: object) -> None:
        """Initialize the JAXFit object and function. This is only done if
        JAX is installed and SciPy length is not None.

        Args:
               multi_func: multi-function object to be used
        """
        if self.scipy_length is not None and JAX is True:
            self.jax = True
            self.jax_func = multi_func(
                self.function_names, self.constraints, use_jax=True
            )
            self.jcf = jaxfit.CurveFit(flength=self.fixed_length)
        else:
            self.jax = False
            self.scipy_length = 0
            # self.jax_func = None

    def get_fit_obj(self, flat_data):
        """Returns the correct fit package to use based on the length of the
        data and whether JAX is installed.

        Args:
            flat_data: flattened data to be fit

        Returns:
            curvefit: curvefit function to use
            fit_func: function to be fit
            print_label: label to print to console if Verbose is enabled
        """
        if self.jax and len(flat_data) > self.scipy_length:
            kwargs = {"return_eval": True}
            return self.jcf.curve_fit, self.jax_func, "JAXFit", True, kwargs
        else:
            return curve_fit, self.func, "SciPy", False, {}

    def get_fit(
        self,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        data: np.ndarray,
        seed: Optional[list[list[float]]] = None,
        bounds: Optional[list[list[float]]] = None,
        sigma: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        uncertainty: bool = False,
        verbose: bool = False,
    ) -> tuple[list[float], dict]:
        """Fit the data to the multi-function

        Args:
            coords: coordinates of the data
            data: data to be fit (must be same shape as coords)
            seed: initial seed for the fit in terms of the multi-functions
                individual functions (ie. list of lists where the top level list
                corresponds to the function and the second level list corresponds
                to the parameters of that function).
            bounds: tuple of min and max values for each parameter, but the
            min and max values are each formatted as a list of lists (see
            seed for formatting)
            sigma: standard deviation of the data (must be same shape as data)
                or a covariance matrix with each axis the same length as
                the data.
            mask: mask to be applied to the data which is the same shape as the
                data
            uncertainty: whether to return the uncertainties of the fit
            plot_it: whether to plot the fit
            verbose: whether to print the fit information to the console.

        Returns:
            params: list of the fit parameters
            info: dictionary of the fit information

        """

        flat_coords, flat_data = self.flatten_fit_data(coords, data)
        if mask is not None:
            mask = mask.flatten()
            flat_coords, flat_data = fit_utils.get_masked_data(
                flat_coords, flat_data, mask
            )

        data_sum = np.sum(flat_data)
        fit_settings = self.get_fit_obj(flat_data)
        curvefit, fit_func, print_label, jaxfit, kwargs = fit_settings

        if seed is None:
            seed = self.get_default_seed(flat_coords, flat_data)
        if bounds is None:
            bounds = self.get_default_bounds()

        # convert the seed and bounds to a single list matching the fit func
        seed = self.func.params_to_args(seed)
        bounds = [self.func.params_to_args(bound) for bound in bounds]
        st = time.time()
        fit_results = curvefit(
            fit_func.fit_function,
            flat_coords,
            flat_data,
            p0=seed,
            bounds=bounds,
            sigma=sigma,
            max_nfev=self.max_nfev,
            **kwargs,
        )

        if jaxfit:
            popt, pcov, func_eval = fit_results
        else:
            popt, pcov = fit_results
            func_eval = self.func.fit_function(flat_coords, *popt)

        if verbose:
            print(print_label, time.time() - st)
        params = self.func.args_to_params(popt)
        save_params = self.handle_uncertainty(popt, pcov, params, uncertainty)
        fit_metrics = self.get_fit_metrics(params, func_eval, flat_data, sigma)
        fit_dict = self.get_info_dict(save_params, fit_metrics, data_sum)

        return params, fit_dict

    def flatten_fit_data(
        self, coords: Union[np.ndarray, Iterable[np.ndarray]], data: np.ndarray
    ):
        """Flatten the data and coordinates to be fit.

        Args:
            coords: coordinates of the data
            data: data to be fit (must be same shape as coords)

        Returns:
            flat_coords: flattened coordinates of the data
            flat_data: flattened data to be fit
        """
        flat_data = data.flatten()
        if type(coords) is tuple or type(coords) is list:
            flat_coords = [coord_array.flatten() for coord_array in coords]
        else:
            flat_coords = coords.flatten()
        return flat_coords, flat_data

    def handle_uncertainty(
        self,
        popt: np.ndarray,
        pcov: np.ndarray,
        func_params: list[list[float]],
        uncertainty: bool,
    ) -> list[list[float]]:
        """Handle the uncertainty of the fit. If uncertainty is True, then
        the covariance matrix and the fit parameters are used to create the
        uncertainty fit parameters. The packages is designed to handle
        these uncertainty parameters throughout, but these are more difficult
        to work with due to limits on the accepted operations and thus the
        user might wish to neglect using them for their own custom functions.

        Args:
            popt: fit parameters
            pcov: covariance matrix for the fit parameters
            func_params: fit parameters in the format list of individual
                functions fit parameters
            uncertainty: whether to return the uncertainty parameters

        Returns:
            save_params: fit parameters to be saved in the info dictionary
        """

        if uncertainty:
            upopt = uncertainties.correlated_values(popt, pcov)
            ufunc_params = self.func.args_to_params(upopt)
            return ufunc_params
        else:
            return func_params

    def get_fit_metrics(
        self,
        params: list[list[float]],
        func_eval: np.ndarray,
        flat_data: Union[np.ndarray, Iterable[np.ndarray]],
        sigma: Optional[np.ndarray] = None,
    ) -> dict:
        """Get the fit metrics for the fit. Currently only chi squared and
        reduced chi squared are calculated.

        Args:
            params: fit parameters
            flat_coords: flattened coordinates of the data
            flat_data: flattened data that was fit
            sigma: standard deviation of the data (must be same shape as data)
                or a covariance matrix with each axis the same length as
                the data.

        Returns:
            fit_metrics: dictionary of the fit metrics
        """
        # TODO: change function evaluation to use JAX rather than numpy
        # current issues is that coords are in numpy and need to be reconverted
        fit_metric_dict = {}
        chi_values = calc_chi_squared(len(params), flat_data, func_eval, sigma)
        chi_dict = {"chi_squared": chi_values[0], "chi_squared_red": chi_values[1]}
        fit_metric_dict.update(chi_dict)
        return fit_metric_dict

    def get_info_dict(
        self, fit_parameters: list[list[float]], fit_metrics: dict, data_sum: float
    ) -> dict:
        """Get the fit information dictionary this will be saved and used
        throughout the package to do things like plotting, integrating, etc.

        Args:
            fit_parameters: fit parameters
            fit_metrics: dictionary of the fit metrics
            data_sum: sum of the data that was fit

        Returns:
            fit_dict: dictionary of the fit information
        """

        fit_info = [
            self.fit_label,
            self.function_names,
            self.constraints,
            fit_parameters,
            fit_metrics,
            data_sum,
        ]
        fit_dict = dict(zip(self.info_keys, fit_info))
        return fit_dict

    def get_default_bounds(self) -> tuple[list[list[float]], ...]:
        """Get the default bounds for the fit. This is a tuple of two lists
        one for the lower bounds and one for the upper bounds. The lower and
        upper bounds are each returned as a list of lists where
        each sublist is the bounds for a single function.

        Returns:
            bounds: tuple of the lower and upper bounds
        """

        bounds = []
        for function_name in self.function_names:
            function_object = self.func_registry.get(function_name)
            bounds.append(function_object.default_bounds())

        min_bounds = [bound[0] for bound in bounds]
        max_bounds = [bound[1] for bound in bounds]
        return min_bounds, max_bounds

    def get_default_seed(
        self, coords: Union[np.ndarray, Iterable[np.ndarray]], data: np.ndarray
    ) -> list[list[float]]:
        """Get the default seed for the fit. This is a list of lists where
        each sublist is the seed for a single function.

        Args:
            coords: coordinates of the data
            data: data to be fit

        Returns:
            seed: list of the seed parameters
        """

        seed = []
        for function_name in self.function_names:
            function_object = self.func_registry.get(function_name)
            seed.append(function_object.initial_seed(coords, data))
        return seed

    def set_cutoff_length(self, scipy_length: int) -> None:
        """Allows the user to change the length of data that will
        trigger a JAX vs. SciPy fit (see init docstring for more info).

        Args:
            scipy_length: length of data to use scipy over jax
        """
        self.scipy_length = scipy_length

    def set_fixed_length(self, fixed_length: int) -> None:
        """Set a fixed length, but jaxfit needs to be changed to allow
        this to work."""
        # self.fixed_length = fixed_length
        pass
