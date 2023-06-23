# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:51:16 2022

@author: hofer
"""
# general python imports
from typing import Iterable, Optional, Union

import numpy as np

# package imports
from atomcloud.fits.func_fits import MultiFit1D
from atomcloud.functions import FUNCTIONS1D, SUMFUNCTIONS
from atomcloud.utils import fit_utils, img_utils
from atomcloud.utils.uncertain_utils import nominal_list


# __all__ = ["SumFit2D"]


class SumFit2D:
    """
    This object sums the image data on the x and y axes and then x and y
    fits are done on the 1D sums. The x and y fits are then used to
    construct a 2D function parameters. The functions passed in must be defined
    in both thev1D function registry as well as the sumfunction registry as
    both sets of objects are used in the fitting process.
    """

    def __init__(
        self,
        function_names: list[str],
        constraints: Optional[list[str]] = None,
        scipy_length: int = 10e3,
        fixed_length: Optional[int] = None,
    ) -> None:
        """
        Initializes the Cloud2DSumFit object which means initializing the
        Cloud1DFit objects for the x and y axes as well as the plotting
        object.

        Args:
            function_names: functions to combined for fit
            constraints: function argument fitting constraints
            scipy_length: integer length of data below which scipy is used
            fixed_length: integer length of data used to initialize jaxfit objs
        """
        super().__init__()
        self.function_names = function_names
        self.constraints = constraints
        self.num_funcs = len(self.function_names)

        self.xmulti_fit1d = MultiFit1D(
            function_names, constraints, scipy_length, fixed_length
        )
        self.ymulti_fit1d = MultiFit1D(
            function_names, constraints, scipy_length, fixed_length
        )

        self.funcs1d_registry = FUNCTIONS1D
        self.sum_registry = SUMFUNCTIONS

        self.info_keys_2d = [
            "fit_type",
            "equations",
            "constraints",
            "params",
            "data_sum",
        ]
        self.fit_dict_keys = ["2d", "xsum", "ysum"]

    # need to properly implement mask below
    def get_fit(
        self,
        coords: Iterable[np.ndarray],
        data: np.ndarray,
        seed: Optional[list[list[float]]] = None,
        bounds: Optional[list[list[float]]] = None,
        mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> tuple[tuple[list[float], ...], dict]:
        """
        Sums the image data on the x and y axes and then x and y fits are
        done on the 1D sums. The x and y fits are then used to construct a
        2D fit params. The x and y fits and 2d params are returned as
        dictionaries.

        Args:
            coords: x and y coordinates of image data
            data: image data to be fit
            seed: initial guess for fit parameters in 2d format?
            bounds: bounds for fit parameters in 2d format?
            mask: mask for image data
            verbose: boolean to print out fit info

        Returns:
            fit_params: x sum fit parameters, y sum fit parameters, 2d fit
                        parameters in a list
            fit_dicts: dictionary with x sum fit info, y sum fit info and
                          2d fit info

        """
        # converts the 2D coordinate and image data to 1D sums
        #  on each axis and 1D coordinates
        x, y, xsum, ysum = img_utils.img_data_to_sums(coords, data, mask)
        xseeds, yseeds = self.get_seed(seed, coords, data)
        xbounds, ybounds = self.get_bounds(bounds, data.shape)
        xparams, xfit_dicts = self.xmulti_fit1d.get_fit(
            x, xsum, xseeds, bounds=xbounds, verbose=verbose
        )
        yparams, yfit_dicts = self.ymulti_fit1d.get_fit(
            y, ysum, yseeds, bounds=ybounds, verbose=verbose
        )

        params_2d = self.convert_all_sums_2d(
            coords, xfit_dicts["params"], yfit_dicts["params"]
        )
        data_sum = (xfit_dicts["data_sum"] + yfit_dicts["data_sum"]) / 2
        fit_dicts = self.get_fit_dicts(params_2d, xfit_dicts, yfit_dicts, data_sum)

        params_2d = nominal_list(params_2d)
        fit_params = (xparams, yparams, params_2d)
        return fit_params, fit_dicts

    def get_fit_dicts(
        self,
        params_2d: list[list[float]],
        xfit_dicts: dict,
        yfit_dicts: dict,
        data_sum: float,
    ) -> dict:
        """
        Creates a dictionary with 2d fit parameters, as well as the x and y
        sum fit parameters.
        Args:
            params_2d: 2D fit parameters
            xfit_dicts: dictionary with x sum fit parameters
            yfit_dicts: dictionary with y sum fit parameters
            data_sum: The average of the x and y sum data.

        Returns:
            all_fit_dicts: dictionary with 2D fit parameters, as well as the
            x and y sum fit parameters.

        """
        values_2d = [
            "cloud2d",
            self.function_names,
            self.constraints,
            params_2d,
            data_sum,
        ]
        fit_dict_2d = dict(zip(self.info_keys_2d, values_2d))
        fit_dict_list = [fit_dict_2d, xfit_dicts, yfit_dicts]
        all_fit_dict = dict(zip(self.fit_dict_keys, fit_dict_list))
        return all_fit_dict

    # is_nested_list

    def is_axes_params(
        self,
        seeds: Union[list[list[float]], list[list[list[float]]]],
    ) -> bool:
        """
        Checks if the seeds are in the 2D format or the 1D format.

        Args:
            seeds: seeds for the fit parameters either in 2D format or 1D
                     format.

        Returns:
            is_axes_params: boolean that is True if the seeds are in the 1D
                            format and False if the seeds are in the 2D format.

        """
        if fit_utils.is_nested_list(seeds, 0) and fit_utils.is_nested_list(seeds, 1):
            return fit_utils.is_nested_list(seeds, 2)
        else:
            raise ValueError("seed input type not recognized")

    # def is_axes_params(
    #     self,
    #     seeds: Union[list[list[float]], list[list[list[float]]]],
    # ) -> bool:
    #     """
    #     Checks if the seeds are in the 2D format or the 1D format.

    #     Args:
    #         seeds: seeds for the fit parameters either in 2D format or 1D
    #                  format.

    #     Returns:
    #         is_axes_params: boolean that is True if the seeds are in the 1D
    #                         format and False if the seeds are in the 2D format.

    #     """
    #     seeds_bool = [isinstance(seed, list) for seed in seeds]
    #     if isinstance(seeds, list) and np.all(seeds_bool):
    #         list_bool = [isinstance(s, list) for seed in seeds for s in seeds]
    #         return np.all(list_bool)
    #     else:
    #         raise ValueError("seed input type not recognized")

    def get_xy_params(
        self,
        coords: Iterable[np.ndarray],
        params: Union[list[list[float]], list[list[list[float]]]],
    ) -> tuple[list[list[float]], ...]:
        """
        Converts the 2D fit parameters to the 1D fit parameters if they
        are in the 2D format.

        Args:
            coords: x and y coordinates of image data
            params: parameters either in 2D format or 1D format

        Returns:
            xparams: x axis parameters in 1D format
            yparams: y axis parameters in 1D format

        """
        if self.is_axes_params(params):  # seed is a list of axis seeds
            xparams, yparams = params
        else:  # seed is 2D seed
            xparams, yparams = self.convert_2d_params_to_1d(coords, params)

        return xparams, yparams

    def get_seed(
        self,
        seed: Union[None, list[list[float]], list[list[list[float]]]],
        coords: Iterable[np.ndarray],
        data: np.ndarray,
    ) -> tuple[list[list[float]], ...]:
        """
        Converts the 2D seed parameters to the 1D seed parameters if they
        are in the 2D format or if they are None then it generates the
        1D seed parameters based on image data and coords

        Args:
            seed: initial guess for fit parameters which is either None or
                in the 2D format or 1D format.
            coords: x and y coordinates of image data
            data: image data to be fit

        Returns:
            xseeds: x axis seeds in 1D format
            yseeds: y axis seeds in 1D format

        """
        if seed is None:
            return self.get_initial_seed(coords, data)
        else:
            return self.get_xy_params(coords, seed)

    def get_initial_seed(
        self,
        coords: Iterable[np.ndarray],
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[list[list[float]], ...]:
        """
        Generates the 1D seed parameters based on image data and coords.
        However just using the initial seed functions from each of the
        respective 1d function classes.

        Args:
            coords: x and y coordinates of image data
            data: image data to be fit

        Returns:
            xseeds: x axis seeds in 1D format
            yseeds: y axis seeds in 1D format

        """
        x, y, xsum, ysum = img_utils.img_data_to_sums(coords, data, mask)
        xseeds = []
        yseeds = []
        for function_name in self.function_names:
            func_obj = self.funcs1d_registry.get(function_name)
            xseeds.append(func_obj.initial_seed(x, xsum))
            yseeds.append(func_obj.initial_seed(y, ysum))
        return xseeds, yseeds

    def get_bounds(self, bounds, data_shape):
        """Bounds needs more work, but basically passes None to both of the
        1D stage fits which operate on the x and y axes respectively."""
        if bounds is None:
            bounds = (None, None)
        return bounds

    def convert_all_sums_2d(
        self,
        coords: Iterable[np.ndarray],
        xparams: list[list[float]],
        yparams: list[list[float]],
    ) -> list[list[float]]:
        """
        Converts the 1D fit parameters for the x and y axes to the 2D fit
        parameters.
        Args:
            coords: x and y coordinates of image data
            xparams: x axis fit parameters in 1D format
            yparams: y axis fit parameters in 1D format

        Returns:
            params_2d: 2D fit parameters for all fit functions
        """
        all_2d_params = []
        for xpar, ypar, name in zip(xparams, yparams, self.function_names):
            func_obj = self.sum_registry.get(name)
            all_2d_params.append(func_obj.convert_sum_2D(coords, xpar, ypar))
        return all_2d_params

    def convert_2d_params_to_1d(
        self, coords: Iterable[np.ndarray], params2d: list[list[float]]
    ) -> tuple[list[list[float]], ...]:
        """
        Converts the 2D fit parameters to the 1D fit parameters along the x
        and y axes.
        Args:
            coords: x and y coordinates of image data
            params2d: 2D fit parameters for all fit functions

        Returns:
            xparams: x axis parameters in 1D format for all fit functions
            yparams: y axis parameters in 1D format for all fit functions
        """
        params1d = []
        for param, function_name in zip(params2d, self.function_names):
            func_obj = self.sum_registry.get(function_name)
            params1d.append(func_obj.convert_2d_sum(coords, param))
        xparams = [param[0] for param in params1d]
        yparams = [param[1] for param in params1d]
        return xparams, yparams
