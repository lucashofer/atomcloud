try:
    from jax.config import config

    config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:
    jnp = None

from abc import ABC, abstractmethod
from inspect import signature
from typing import Iterable, Union

import numpy as np


# __all__ = ["FunctionBase"]


class FunctionBase(ABC):
    """Base class for function objects"""

    def __init__(self):
        """Instantiates the function object, the only argument is whether
        to use jax for the function"""
        self.function = self.make_function(use_jax=False)
        self.create_parameter_dict()

    @abstractmethod
    def create_function(self, anp: object) -> callable:
        """
        Creates the function using the numpy or jax object given. This method
        must be overridden by the child class.
        Args:
            anp: The numpy or jax object to use for the function

        Returns:
            The created function
        """
        pass

    def __call__(
        self, coords: Union[np.ndarray, Iterable[np.ndarray]], *params: float
    ) -> np.ndarray:
        """
        Calls the function with the given coordinates and parameters and
        is the default function to be called if the object is called as a
        function.
        Args:
            coords: The coordinates to evaluate the function at
            *params: The parameters of the function

        Returns:
            The value of the function at the given coordinates
        """
        return self.function(coords, *params)

    def make_function(self, use_jax: bool = False) -> callable:
        """
        Creates the class function. This function is
        created using the create_function method and will use jax if use_jax is
        True otherwise numpy will be used.

        Args:
            use_jax: Whether or not to use jax for the function.

        Returns:
            The jax or numpy function that is created

        """
        if use_jax:
            if jnp is not None:
                return self.create_function(jnp)
            else:
                raise Exception("JAX/JAXFit is not installed")
        else:
            return self.create_function(np)

    def create_parameter_dict(self) -> None:
        """Creates a dictionary of the parameters of the function"""
        self.param_dict = list(signature(self.function).parameters)[1:]

    def analyze_parameters(self, params: list[float]) -> dict:
        """
        Analyzes the fit parameters of the function and returns a dictionary of
        the analysis parameters.
        Args:
            params: The function parameters determined by the fit
        Returns:
            A dictionary of the analysis parameters
        """
        return {}

    def rescale_parameters(self, params: list[float], scales: list) -> list[float]:
        """
        Rescales the parameters of the function determined by the fit by
        the scales given for the x y and z axes.
        Args:
            params: The parameters of the function determined by the fit
            scales: The scales for the x y and z axes

        Returns:
            The rescaled fit parameters
        """
        return params

    def rescale_analysis_params(self, params: dict, scales: list) -> dict:
        """
        Rescales the analysis parameters constructed from the fit parameters
        by the scales given for the x y and z axes.
        Args:
            params: The analysis parameters constructed from the fit parameters
            scales: The scales for the x y and z axes

        Returns:
            The rescaled analysis parameters

        """
        return params

    def initial_seed(
        self, coords: Union[np.ndarray, Iterable[np.ndarray]], data: np.ndarray
    ) -> list[float]:
        """
        Returns the initial seed parameters for the fit. The default is to
        return a list of ones for the parameters. This method can be overridden
        to return a different initial seed if something more intelligent is
        desired.
        Args:
            coords: The coordinates to fit the function to
            data: The data to fit the function to

        Returns:
            The initial seed parameters for the fit

        """
        return [1.0 for _ in self.param_dict]

    def default_bounds(self) -> tuple[list[float], ...]:
        """
        Returns the default bounds for the fit. The default is to return
        (-np.inf, np.inf) for all parameters. This method can be overridden
        to return a different set of bounds if something more intelligent is
        desired.
        Returns:
            The default bounds for the fit

        """
        min_bounds = [-np.inf for _ in self.param_dict]
        max_bounds = [np.inf for _ in self.param_dict]
        return (min_bounds, max_bounds)
