# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 12:09:07 2022

@author: hofer
"""
import copy
from typing import Optional

from atomcloud.functions import FUNCTIONS1D, FUNCTIONS2D
from atomcloud.process_fits.base import type_fitdict
from atomcloud.process_fits.iterate import IterateFitDict


# __all__ = ["analyze_fitdict", "analyze_1d_fit", "analyze_2d_fit"]


def parameter_analysis(fit_dict: dict, func_registry: object) -> dict:
    """
    Analyze the fit parameters from a multi-function fit. The analysis
    functions themselves are defined by the individual function objects in
    atomcloud.functions.funcs_1d or atomcloud.functions.funcs_2d. By default
    these include integrating the fitted function and calculating the
    lab-frame widths for the 2d fits.

    Args:
        fit_dict: dictionary of fit results and info
        func_registry: registry of either 1d or 2d function objects

    Returns:
        The input fit dict, but with the analysis parameters added

    """
    equations = fit_dict["equations"]
    params = fit_dict["params"]
    analysis_params = []
    for pars, equation in zip(params, equations):
        func_obj = func_registry.get(equation)
        analysis_param_func = func_obj.analyze_parameters
        analysis_params.append(analysis_param_func(pars))
    fit_dict["analysis_params"] = analysis_params
    return fit_dict


def analyze_1d_fit(fit_dict: dict) -> dict:
    """Caculates the analysis parameters for a 1d cloud multi-function fit.
    using the function objects in atomcloud.functions.funcs_1d

    Args:
        fit_dict: dictionary of fit results and info

    Returns:
        The input fit dict, but with the analysis parameters added
    """
    fit_dict = copy.deepcopy(fit_dict)
    func_registry = FUNCTIONS1D
    return parameter_analysis(fit_dict, func_registry)


def analyze_2d_fit(fit_dict: dict) -> dict:
    """Caculates the analysis parameters for a 2d multi-function fit.
    using the function objects in atomcloud.functions.funcs_2d.

    Args:
        fit_dict: dictionary of fit results and info

    Returns:
        The input fit dict, but with the analysis parameters added
    """
    fit_dict = copy.deepcopy(fit_dict)
    func_registry = FUNCTIONS2D
    return parameter_analysis(fit_dict, func_registry)


class AnalyzeFitDicts(IterateFitDict):
    """Class to analyze the fit parameters from a multi-function fit or series
    of multi-function fits.
    """

    def __init__(self):
        """See IterateFitDict for more info."""
        super().__init__()

    def process_fitdict1d(self, fit_dict: dict, *args, **kwargs) -> dict:
        """Calls the analyze_1d_fit function to analyze the fit parameters,
        but within the context of the IterateFitDict class. This allows the
        analysis to be applied to each level of a multi-level fit dict.
        """
        return analyze_1d_fit(fit_dict)

    def process_fitdict2d(self, fit_dict: dict, *args, **kwargs) -> dict:
        """Calls the analyze_2d_fit function to analyze the fit parameters,
        but within the context of the IterateFitDict class. This allows the
        analysis to be applied to each level of a multi-level fit dict.
        """
        return analyze_2d_fit(fit_dict)

    def analyze_fitdict(self, fit_dict: dict, dict_type: Optional[str] = None) -> dict:
        """Analyze the fit parameters from a multi-function fit or series
        of multi-function fits. If a multi-level fit dict is passed, the
        analysis will be applied to each level of the dict using the parent
        class IterateFitDict. The analysis functions themselves are defined
        by the individual function objects in atomcloud.functions.funcs_1d or
        atomcloud.functions.funcs_2d.

        Args:
            fit_dict: fit dictionary to be analyzed
            dict_type: type of fit dictionary. If not given, the type will
                be inferred from the fit dict.

        Returns:
            analyzed fit dictionary
        """

        if dict_type is None:
            dict_type = type_fitdict(fit_dict)
        analysis_func = self.fit_type_dict[dict_type]
        return analysis_func(fit_dict)


def analyze_fitdict(fit_dicts: dict, dict_type: Optional[str] = None) -> dict:
    """Initialize the AnalyzeFitDicts class and call the analyze_fitdict
    method. This allows any fit dict to be analyzed without having to
    initialize the class. The analysis functions themselves are defined
    by the individual function objects in atomcloud.functions.funcs_1d or
    atomcloud.functions.funcs_2d.

    Args:
        fit_dicts: fit dictionary to be analyzed
        dict_type: type of fit dictionary. If not given, the type will
            be inferred from the fit dict.

    Returns:
        analyzed fit dictionary
    """
    afd = AnalyzeFitDicts()
    return afd.analyze_fitdict(fit_dicts, dict_type)
