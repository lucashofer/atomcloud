# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:16:36 2022

@author: hofer
"""
from typing import Optional

from atomcloud.functions import funcs_1d, funcs_2d
from atomcloud.functions.multi_base import ConstrainedMultiFunction


# __all__ = ["MultiFunc", "MultiFunction1D", "MultiFunction2D"]


class MultiFunc(ConstrainedMultiFunction):
    """Base class for 1D and 2D cloud multi-function classes which itself
    inherits from the ConstrainedMultiFunction class. This function allows
    the user to combine multiple functions together into a single function
    which can then be used in SciPy or JAXFit curve fitting functions.
    Can also include constraints between the parameters of the functions."""

    def __init__(
        self,
        function_names: list[str],
        func_registry: object,
        constraints: Optional[list[str]] = None,
        use_jax: bool = False,
    ) -> None:

        """Initialize the multi-function object which will combine multiple
        functions into a single function.

        Args:
            function_names: The keys for the function objects in the registry
                which will be used in the multi-function
            func_registry: The registry of function objects
            constraints: A list of constraints which will be applied to the
                the functions in the multi-function (see ConstrainedMultiFunction
                for more details)
            use_jax: If True, the functions in the multi-function will be
                created using JAX. If False, the functions will be created

        Returns:
            None
        """
        functions = []
        for key in function_names:
            func_obj = func_registry.get(key)
            if use_jax:  # construct unique jax function to avoid retracing
                functions.append(func_obj.make_function(use_jax))
            else:
                functions.append(func_obj.function)  # np func already defined
        super().__init__(functions, constraints)


class MultiFunction2D(MultiFunc):
    """2D cloud multi-function class which inherits from the base class.
    It uses the imported dictionary of 2D function objects as it's base
    dictionary of function objects, but also allows the user to add custom
    function objects to the dictionary of function objects.

    Args:
            function_names: The keys for the function objects in the registry
                which will be used in the multi-function
            func_registry: The registry of function objects
            constraints: A list of constraints which will be applied to the
                the functions in the multi-function
                (see ConstrainedMultiFunction
                for more details)
            use_jax: If True, the functions in the multi-function will be
                created using JAX. If False, the functions will be created

    Returns:
            None
    """

    def __init__(
        self,
        function_names: list[str],
        constraints: Optional[list[str]] = None,
        use_jax: bool = False,
    ) -> None:

        func_registry = funcs_2d.FUNCTIONS2D
        super().__init__(function_names, func_registry, constraints, use_jax)


class MultiFunction1D(MultiFunc):
    """1D cloud multi-function class which inherits from the base class.
    It uses the imported dictionary of 2D function objects as it's base
    dictionary of function objects, but also allows the user to add custom
    function objects to the dictionary of function objects.

    See base class for more details."""

    def __init__(
        self,
        function_names: list[str],
        constraints: Optional[list[str]] = None,
        use_jax: bool = False,
    ) -> None:

        func_registry = funcs_1d.FUNCTIONS1D
        super().__init__(function_names, func_registry, constraints, use_jax)
