# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:49:49 2022

@author: hofer
"""
import copy
import warnings
from abc import ABC
from inspect import signature
from typing import Iterable, Optional, Union

import numpy as np


# TODO constrain function as well as fit_function for consistency
# __all__ = ["MultiFunctionBase", "ConstrainedMultiFunction"]


class MultiFunctionBase(ABC):
    """
    Class which allows multiple functions to be added together to form a
    single function. This is useful for multi function fitting.
    """

    def __init__(self, funcs: list[callable]) -> None:
        """
        Initialize the multi function object with a list of functions to be
        added together.

        Args:
            funcs: list of functions to be added together
        """
        self.num_funcs = len(funcs)
        self.functions = funcs
        # self.construct_multi_function()

    # def construct_multi_function(self):
    #     """
    #     Creation function for the multi function. We use the creation function
    #     so that self is not passed to the multi function. This is because
    #     JAXFIT does not allow self to be passed to the function due JIT
    #     compilation.

    #     """

    #     def function(
    #         coords: Union[np.ndarray, Iterable[np.ndarray]],
    #         function_parameters: list[list[float]],
    #     ) -> np.ndarray:
    #         """
    #         Multi function which adds together the functions passed to the
    #         class initializer.

    #         Args:
    #             coords: coordinates of the data
    #             function_parameters: list of parameters for each function

    #         Returns:
    #             The value of the multi function at the given coordinates

    #         """
    #         func_data = 0
    #         for func, params in zip(self.functions, function_parameters):
    #             func_data = func_data + func(coords, *params)
    #         return func_data

    #     self.function = function

    def function(
        self,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        function_parameters: list[list[float]],
    ) -> np.ndarray:
        """
        Multi function which adds together the functions passed to the
        class initializer.

        Args:
            coords: coordinates of the data
            function_parameters: list of parameters for each function

        Returns:
            The value of the multi function at the given coordinates

        """
        func_data = 0
        for func, params in zip(self.functions, function_parameters):
            func_data = func_data + func(coords, *params)
        return func_data

    def __call__(
        self,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        function_parameters: list[list[float]],
    ) -> np.ndarray:
        """
        Calls the multi function with the given coordinates and parameters and
        is the default function to be called if the object is called as a
        function."""
        return self.function(coords, function_parameters)


class ConstrainedMultiFunction(MultiFunctionBase):
    """
    Creates a multi function which takes a list of functions and a list of
    constraints. The constraints are the input arguments which should have
    the same value when in a sub-functions parameters. The individual function
    parameters are then mapped to a single list of arguments which are passed
    to the constrained multi function.

    The fit function arguments are then passed to the multi
    function which calls the functions with the correct parameters.
    """

    def __init__(
        self, funcs: list[callable], constraints: Optional[list[str]] = None
    ) -> None:
        """
        Initialize the constrained multi function object with a list of
        functions to be added together and a list of constraints.

        Args:
            funcs: list of functions to be added together
            constraints: list of constraints (input parameters)
            which should be constrained when found in in any of the
            individual functions input parameters.
        """
        super().__init__(funcs)
        if constraints is None:
            constraints = []
        funcs_args = [list(signature(func).parameters)[1:] for func in funcs]
        self.param_lengths = [len(args) for args in funcs_args]
        # uniques arguments from all function input arguments
        unique_args = list(set([arg for slist in funcs_args for arg in slist]))
        self.check_contraint_strs(constraints, unique_args)
        # get constraints that are actual function arguments
        used_constraints = list(set(constraints).intersection(unique_args))
        # map from 2d func params to 1d fit func args
        self.arg_param_map = self.get_arg_map(used_constraints, funcs_args)
        # now map the opposite way
        self.param_arg_map = self.arg_list(self.arg_param_map, funcs_args)
        self.create_args_to_params()  # need creator function for JAX class func

        # self.func_args = funcs_args
        self.num_args = len(self.param_arg_map)

    def get_arg_map(
        self, constraints: list[str], func_args: list[list[str]]
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Creates a dictionary which maps the constrained arguments to the
        individual function parameters.
        Args:
            constraints: list of constraints (input parameters)
            func_args: list of lists of individual function input arguments
        Returns:
            Dictionary which maps the index of the single argument list for
            the scipy/jaxfit compatible fit function to the two indices of the
            individual function parameters. The first index is the index of
            the function in the list of functions and the second index is the
            index of the parameter in the list of parameters for that function.

        """
        constraint_map = {constr: i for i, constr in enumerate(constraints)}
        arg_param_map = {i: [] for i, constr in enumerate(constraints)}
        count = len(constraint_map)
        for ind1, sublist in enumerate(func_args):
            for ind2, arg in enumerate(sublist):
                if arg in constraints:
                    con_ind = constraint_map[arg]
                    arg_param_map[con_ind].append((ind1, ind2))
                else:
                    arg_param_map[count] = [(ind1, ind2)]
                    count += 1
        return arg_param_map

    def arg_list(
        self,
        arg_param_map: dict[str, list[tuple[int, int]]],
        funcs_args: list[list[str]],
    ) -> list[list[int]]:
        """Opposite to above, makes a list of lists of the same size as the
        parameters. Each entry is the index of that variable in the single
        argument list for the scipy/jaxfit compatible fit function

        Args:
            arg_param_map: See return of get_arg_map function for detailed
                description of this argument.
            funcs_args: all arguments of the individual functions excluding
                the first coordinate argument

        Returns:
            List of lists of the same size as the parameters. Each entry is
            the index of that variable in the 1D

        """
        param_arg_map = [[np.nan for _ in fargs] for fargs in funcs_args]
        for ind in range(len(arg_param_map)):
            for (i1, i2) in arg_param_map[ind]:
                param_arg_map[i1][i2] = ind
        return param_arg_map

    def __call__(
        self,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        function_parameters: list[list[float]],
    ) -> np.ndarray:
        """
        Overwrite the call function from the parent class to consrtain
        the parameters before calling the function."""
        func_params = copy.deepcopy(function_parameters)
        for key, arg_list in self.arg_param_map.items():
            i10, i20 = arg_list[0]
            for (i1, i2) in arg_list[1:]:
                func_params[i1][i2] = func_params[i10][i20]
        return self.function(coords, func_params)

    def fit_function(
        self, coords: Union[np.ndarray, Iterable[np.ndarray]], *args: float
    ) -> np.ndarray:
        """The fit function which takes a list of arguments and converts them
        to the 2D list of function parameters and then call the
        multi-function. This is compatible with the scipy/jaxfit curvefit
        function input type which requires a single argument list.

        Args:
            coords: coordinates of the data being fit
            *args: list of multi-function arguments which are converted then
            converted to parameters for the individual functions comprising
            the multi-function.

        Returns:
            The value of the multi-function at the given coordinates.

        """
        function_parameters = self.args_to_params(args)
        return self.function(coords, function_parameters)

    def create_args_to_params(self):
        """Creates a function which maps the 1D list of fit function arguments
        to the 2D list of function parameters. We use a creator function
        wrapper so that the args_to_params function can be used in a JAX
        (i.e. to self argument)"""

        def args_to_params(args: list[float]) -> list[list[float]]:
            """
            Maps the SciPy/JAX compatible fit function arguments to the
            individual function parameters for those functions which comprise
            the multi-function.
            Args:
                args: list of arguments for the fit function

            Returns:
                List of lists of function parameters for the individual
                functions comprising the multi-function.

            """
            function_parameters = []
            for arg_list in self.param_arg_map:
                function_parameters.append([args[i] for i in arg_list])
            return function_parameters

        self.args_to_params = args_to_params

    def params_to_args(self, params: list[list[float]]) -> list[float]:
        """Converts the list of lists of individual function parameters
        into a 1D list of arguments for the fit function compatible with
        SciPy/JAX curvefit.

        Args:
            params: list of lists of function parameters for the individual
                functions comprising the multi-function.

        Returns:
            Single list of arguments for the fit function which is compatible
            with the SciPy/JAX curvefit function.

        """
        self.check_parameters(params)
        args = []
        for key, tuple_list in self.arg_param_map.items():
            ind1, ind2 = tuple_list[0]
            args.append(params[ind1][ind2])
        return args

    def check_parameters(self, aparams: list[list[float]]) -> None:
        """Checks that the list of list of func parameters is the correct
        type and that each sublist is the correct length.

        Args:
            aparams: list of lists of function parameters for the individual
                functions comprising the multi-function.
        """
        if not (isinstance(aparams, list) or isinstance(aparams, np.ndarray)):
            raise TypeError("Parameters must be a list of lists")

        if not self.num_funcs == len(aparams):
            raise TypeError(
                "Number of functions does not match number of " "parameter lists"
            )

        for index, params in enumerate(aparams):
            if isinstance(params, list) or isinstance(params, np.ndarray):
                num_func_params = self.param_lengths[index]
                num_params = len(params)
                if num_func_params != num_params:
                    raise ValueError(
                        f"""Parameter list {index} should have 
                        length {num_func_params}, but currently is length
                        {num_params}"""
                    )
            else:
                raise TypeError(
                    """Parameters is currently only a list. It needs 
                    to be a list of lists"""
                )

    def check_contraint_strs(
        self, constraints: list[str], unique_args: list[str]
    ) -> None:
        """Checks that the constraints are valid strings and that they are
        arguments of the functions.

        Args:
            constraints: list of strings which are the names of the
                parameters which are constrained to between the individual
                functions comprising the multi-function.
            unique_args: the unique input arguments across all the individual
                functions comprising the multi-function.

        """
        if not isinstance(constraints, list):
            raise TypeError(
                "Contraint input for a single fit stage must be a list of strings."
            )

        for constraint in constraints:
            if isinstance(constraint, str):
                if constraint not in unique_args:
                    warnings.warn(f"{constraint} is not in any function used.")
            else:
                raise TypeError("Individual constraints must be defined with strings.")
