# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:08:19 2022

@author: hofer
"""
# exterior imports
import copy
from typing import Iterable, Optional, Union

import numpy as np

from atomcloud.fits import MultiFit1D, MultiFit2D, SumFit2D


# __all__ = ["MixedLevelBase"]

FIT_TYPE_DICT = {"1DFit": MultiFit1D, "2DFit": MultiFit2D, "SumFit2D": SumFit2D}

# TODO: add bounds to mixed level fit


def fit_level(
    fit_type: str, function_names: list[str], constraints: Optional[list[str]] = None
):
    """Decorator to construct a fit level for a mixed level fit.

    This decorator will construct a fit level for a mixed level fit. The function
    that is decorated will be used to pre-process the data before it is passed
    to the fit level. The name of the decorated function will be used as the
    name of the fit level.

    Args:
        fit_type: The type of fit to be used. Must be one of the following
        keys '1DFit', '2DFit', 'SumFit2D'.
        function_names: List of function names to be used in the fit.
        constraints: List of constraints to be used in the fit.

    Returns:
        A decorator that will construct a fit level for a mixed level fit.

    Raises:
        ValueError: If fit_type is not one of the keys in FIT_TYPE_DICT.
    """

    def construct_fit_level(pre_func):
        level_name = pre_func.__name__

        def wrapper(self, *args, **kwargs):
            if fit_type in FIT_TYPE_DICT.keys():
                fit_class = FIT_TYPE_DICT[fit_type]
                fit_obj = fit_class(
                    function_names,
                    constraints,
                    scipy_length=self.scipy_length,
                    fixed_length=self.fixed_length,
                )
            else:
                raise ValueError(f"Fit_type {fit_type} not recognized.")
            self.level_fit_objs[level_name] = fit_obj
            self.level_prefunc[level_name] = pre_func

        return wrapper

    return construct_fit_level


class MixedLevelBase:
    """Base class for mixed level fits."""

    def __init__(
        self,
        scipy_length: int = 1e3,
        fixed_length: Optional[int] = None,
        # save_data: bool = True,
    ):
        """
        This class constructs a mixed level fit. Since the fit is defined by
        the fit_order class variable, which corresponds to the decorated
        functions, this class creates the fit level for each of the functions
        in the fit_order list.

        Args:
            scipy_length: length of data to use scipy over jax
            fixed_length: fixed length in JAXFit
            save_data: Whether or not to save the data and coordinates for each
            fit level.
            verbose: Whether to print fit labels throughout.

        Raises:
            ValueError: If the fit_order list is empty.
            ValueError: If the fit_order list contains the string 'default'.
            ValueError: If the fit_order list contains a string that is not
            a function in the class.
        """
        self.scipy_length = scipy_length
        self.fixed_length = fixed_length
        # self.save_data = save_data

        self.level_fit_objs = {}
        self.level_prefunc = {}

        methods = []
        for func in dir(self):
            if not func.startswith("__"):
                if callable(getattr(self, func)):
                    methods.append(func)

        # check if users fit order names are valid, then call them
        for func_name in self.fit_order:
            if func_name == "default":
                raise ValueError('Function name "default" reserved.')
            elif func_name in methods:
                getattr(self, func_name)()
            else:
                raise ValueError(f"Fit function {func_name} not defined in class.")

    def create_default_dicts(
        self,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        data: np.ndarray,
    ) -> None:
        """Create the default dictionaries for the fit.

        This function will create the dictionaries to store fit, data,
        coordinates, masks and seeds for each fit level. These
        dictionaries are recreated each time a new fit is performed. The
        level_fits dictionary will store the fit dictionaries for each fit
        level and will always be created. The other dictionaries will only be
        created if save_data is True.

        Args:
            coords: The coordinates of the data.
            data: The data to be fit.
        """
        self.level_fits = {}
        if self.save_data:
            self.level_seeds = {}
            self.level_masks = {}
            self.level_data = {"default": data}
            self.level_coords = {"default": coords}
        else:
            self.level_seeds = None
            self.level_masks = None
            self.level_data = None
            self.level_coords = None

    def get_fit(
        self,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        data: np.ndarray,
        save_data: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> tuple[bool, dict]:

        """Get the fit for the mixed level fit.

        This function will get the fit for the mixed level fit. Each fit
        level will be called in the order specified by the fit_order class
        variable. If a fit fails, then the fit will break out of the fit
        sequence and return the fit information up to that point.

        The user can add their own args and kwargs to the input parameters.
        These are then assigned to the self.args and self.kwargs class
        variables, which allows the user to use these parameters in the
        (decorated) pre-processing functions.

        If save data  is True, then the data, coordinates, masks and seeds
        will be saved for each fit level. However, the initial data and
        coordinates will be saved under the key 'default'. If any other fit
        level uses the same data and coordinates, as the default fit data, then
        the data and coordinates will be None.

        Args:
            coords: The coordinates of the data.
            data: The data to be fit.
            save_data: Whether or not to save the data and coordinates
            for each fit level.
            *args: Additional arguments to be passed to the fit.
            **kwargs: Additional keyword arguments to be passed to the fit.

        Returns:
            A tuple containing a boolean that indicates if the fit passed and
            a dictionary of fit dictionaries.
        """

        self.coords = coords
        self.data = data
        self.save_data = save_data
        self.args = args
        self.kwargs = kwargs
        self.create_default_dicts(coords, data)

        for level_name in self.fit_order:
            self.fit_passed = self.get_level_fit(level_name, coords, data, verbose)
            if not self.fit_passed:
                break

        fit_info = {"fitdicts": copy.deepcopy(self.level_fits)}
        fit_info["data"] = self.level_data.copy()
        fit_info["coords"] = copy.deepcopy(self.level_coords)
        fit_info["masks"] = self.level_masks.copy()
        fit_info["seeds"] = copy.deepcopy(self.level_seeds)
        return self.fit_passed, fit_info

    def get_level_fit(self, level_name, coords, data, verbose=False):
        """Get the fit for a single level.

        This function will get the fit for a single level. The pre-processing
        function will be called first, which will return the coordinates,
        data, seed and mask. The fit function will then be called, which will
        return the fit dictionary. The fit dictionary will be stored in the
        level_fits dictionary.

        Args:
            level_name: The name of the fit level.
            coords: The coordinates of the data.
            data: The data to be fit.
            verbose: Whether or not to print the error message if the fit
            fails.

        Returns:
            A boolean that indicates if the fit passed.
        """
        # try:
        pre_func = self.level_prefunc[level_name]
        output = pre_func(self, coords, data)
        coords, data, seed, mask = output
        if self.save_data:
            self.level_seeds[level_name] = seed
            self.level_masks[level_name] = mask
            self.save_fit_data(level_name, data)
            self.save_fit_coords(level_name, coords)
        fit_func = self.level_fit_objs[level_name].get_fit
        _, level_fit_dicts = fit_func(
            coords, data, seed=seed, mask=mask, verbose=verbose
        )
        self.level_fits[level_name] = level_fit_dicts
        return True
        # except Exception as e:
        #     if verbose:
        #         print(level_name, e)
        #     return False

    def save_fit_data(self, level_name: str, data: np.ndarray):
        """Save the fit data for a given level.

        This function will save the fit data for a given level. If the data
        is the same as the initial data, then the data will be set to None.
        This is to save memory, since the initial data is saved under the
        key 'default'.

        Args:
            level_name: The name of the fit level.
            data: The data to be saved.
        """

        if self.save_data:
            if np.all(self.data == data):
                self.level_data[level_name] = None
            else:
                self.level_data[level_name] = data

    def save_fit_coords(
        self, level_name: str, coords: Union[np.ndarray, Iterable[np.ndarray]]
    ):
        """Save the fit coordinates for a given level.

        This function will save the fit coordinates for a given level. If the
        coordinates are the same as the initial coordinates, then the
        coordinates will be set to None. This is to save memory, since the
        initial coordinates are saved under the key 'default'.

        Args:
            level_name: The name of the fit level.
            coords: The coordinates to be saved.
        """
        if isinstance(self.coords, np.ndarray):
            if isinstance(coords, np.ndarray):
                if np.all(self.coords == coords):
                    save = False
            else:
                raise ValueError(
                    "coords must be np.ndarray if self.coords is np.ndarray"
                )

        elif isinstance(self.coords, Iterable):
            same_coord = []
            for coord, init_coord in zip(coords, self.coords):
                if not isinstance(coord, np.ndarray):
                    raise ValueError(
                        "coords must be Iterable[np.ndarray] if  \
                                     self.coords is Iterable[np.ndarray]"
                    )
                else:
                    if np.all(init_coord == coord):
                        same_coord.append(True)
                    else:
                        same_coord.append(False)
            if np.all(same_coord):
                save = False

        if save:
            self.level_coords[level_name] = coords
        else:
            self.level_coords[level_name] = None
