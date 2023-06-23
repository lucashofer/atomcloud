# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:09:20 2022

@author: hofer
"""
from typing import Optional

from atomcloud.fits.multi_fit import MultiFunctionFit
from atomcloud.functions import (
    FUNCTIONS1D,
    FUNCTIONS2D,
    MultiFunction1D,
    MultiFunction2D,
)


# __all__ = ["MultiFit1D", "MultiFit2D"]


class MultiFit1D(MultiFunctionFit):
    """
    Class for doing 1d multi-function fits. Inherits from the base class
    MultiFunctionFit.
    """

    def __init__(
        self,
        function_names: list[str],
        constraints: Optional[list[str]] = None,
        scipy_length: int = 1e3,
        fixed_length: Optional[int] = None,
        max_nfev_scalar: int = 100,
    ):
        """
        Performs a 1D multi-function fit. The user can specify the functions
        to be used in the fit, as well as any constraints between the parameters
        of the functions. The user can also specify whether to use JAX or
        SciPy for the fit.

        Args:
            function_names: The keys for the function objects in the 1d
                function registry which will be used in the multi-function
            constraints: A list of constraints which will be applied to the
                the functions in the multi-function
                (see ConstrainedMultiFunction for more details)
            scipy_length: The length of the data below which the SciPy fit
                will be used instead of the JAXFit. Defaults to 1e3.
            fixed_length: The length of the fixed length data if JAXFit is
                used. Defaults to None.
            max_nfev_scalar: An integer scalar multiplied by the number of
                parameters in the fit to determine the maximum number of
                function evaluations for the SciPy/JAXFit fit. Defaults to 100.
        """
        multi_func = MultiFunction1D
        func_registry = FUNCTIONS1D
        fit_label = "1d"
        super().__init__(
            function_names,
            multi_func,
            func_registry,
            fit_label,
            max_nfev_scalar,
            constraints,
            scipy_length,
            fixed_length,
        )


class MultiFit2D(MultiFunctionFit):
    def __init__(
        self,
        function_names: list[str],
        constraints: Optional[list[str]] = None,
        scipy_length: int = 1e3,
        fixed_length: Optional[int] = None,
        max_nfev_scalar: int = 30,
    ) -> None:

        """Performs a 2D multi-function fit. The user can specify the functions
        to be used in the fit, as well as any constraints between the parameters
        of the functions. The user can also specify whether to use JAX or
        SciPy for the fit.

        Args:
            function_names: The keys for the function objects in the 1d
                function registry which will be used in the multi-function
            constraints: A list of constraints which will be applied to the
                the functions in the multi-function
                (see ConstrainedMultiFunction for more details)
            scipy_length: The length of the data below which the SciPy fit
                will be used instead of the JAXFit. Defaults to 1e3.
            fixed_length: The length of the fixed length data if JAXFit is
                used. Defaults to None.
            max_nfev_scalar: An integer scalar multiplied by the number of
                parameters in the fit to determine the maximum number of
                function evaluations for the SciPy/JAXFit fit. Defaults to 30.
        """

        multi_func = MultiFunction2D
        func_registry = FUNCTIONS2D
        fit_label = "2d"

        super().__init__(
            function_names,
            multi_func,
            func_registry,
            fit_label,
            max_nfev_scalar,
            constraints,
            scipy_length,
            fixed_length,
        )
