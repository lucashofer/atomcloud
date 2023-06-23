import copy
from typing import Optional, Union

from atomcloud.functions import FUNCTIONS1D, FUNCTIONS2D
from atomcloud.process_fits.base import type_fitdict
from atomcloud.process_fits.iterate import IterateFitDict


# __all__ = ['rescale_fitdict', 'rescale_1d_fit', 'rescale_2d_fit']


def get_number_scale(xscale: float, yscale: float, zscale: float) -> float:
    """Scalar to multiple integrated atom numbers by. Takes into account
    scaling on all axes.

    Args:
        xscale: scalar to rescale x parameters by
        yscale: scalar to rescale y parameters by
        zscale: scalar to rescale z parameters by

    Returns:
        number_scale: scalar to multiple integrated atom numbers by
    """
    return xscale * yscale * zscale


def rescale_fit_dict(
    fit_dict: dict, number_scale: float, func_scale: list, func_registry: object
) -> dict:
    """Rescale the parameters of a single multi-function fit dictionary
    including the fit parameters, analysis parameters and data sums.

    Args:
        fit_dict: dictionary for a single multi-function fit
        number_scale: scalar to multiply integrated atom numbers
        func_scale: list of scalars to multiply function parameters by
        func_registry: registry of function objects corresponding to the
            equations used in the fit

    Returns:
        rescale_fit_dict: rescaled fit dictionary
    """

    fit_dict["params"] = rescale_params(
        func_registry, fit_dict["equations"], fit_dict["params"], func_scale
    )
    if "analysis_params" in fit_dict:
        analysis_params = rescale_params(
            func_registry,
            fit_dict["equations"],
            fit_dict["analysis_params"],
            func_scale,
            analysis_params=True,
        )
        fit_dict["analysis_params"] = analysis_params
    fit_dict["data_sum"] = fit_dict["data_sum"] * number_scale
    return fit_dict


def rescale_params(
    func_registry: object,
    equations: list[str],
    params: Union[list[list[float]], list[dict[str, float]]],
    func_scale: float,
    analysis_params: bool = False,
) -> list[float]:
    """Rescales either the fit parameters or analysis parameters for a
    multi-function fit. Iterates through each individual function in the
    multi-function fit and rescales those functions parameters using the
    rescaling equations defined in the individual function objects
    (called from the function registry).

    Args:
        func_registry: registry of function objects corresponding to the
            equations used in the fit
        equations: list of equations used in the fit
        params: list of lists of fit parameters or analysis parameters
        func_scale: list of scalars to multiply function parameters by
        analysis_params: boolean to indicate if the parameters are fit
            parameters or analysis parameters

    Returns:
        rescaled_params: rescaled fit parameters or analysis parameters
    """
    rescaled_params = []
    for pars, equation in zip(params, equations):
        func_obj = func_registry.get(equation)
        if analysis_params:
            rescale_pars = func_obj.rescale_analysis_params(pars, func_scale)
        else:
            rescale_pars = func_obj.rescale_parameters(pars, func_scale)
        rescaled_params.append(rescale_pars)
    return rescaled_params


def rescale_1d_fit(
    fit_dict: dict,
    axis: Optional[str] = None,
    axis_scale: Optional[str] = None,
    xscale: float = 1.0,
    yscale: float = 1.0,
    zscale: float = 1.0,
) -> dict:
    """Rescale a 1d multi-function fit dictionary including the fit parameters,
    analysis parameters and data sums.

    Args:
        fit_dict: 1d multi-function fit dictionary.
        axis: axis to rescale. Must be 'x' or 'y'
        axis_scale: scale to rescale axis by. Must be 'x' or 'y'
        xscale: scalar to rescale x parameters by
        yscale: scalar to rescale y parameters by
        zscale: scalar to rescale z parameters by

    Returns:
        rescale_fit_dict: rescaled fit dictionary
    """

    fit_dict = copy.deepcopy(fit_dict)
    func_registry = FUNCTIONS1D
    if axis_scale is None:
        if axis is None:
            raise ValueError("axis or axis scale must be defined")
        elif axis == "x":
            axis_scale = xscale
        elif axis == "y":
            axis_scale = yscale

    number_scale = xscale * yscale * zscale
    func_scale = [number_scale, axis_scale, zscale]
    return rescale_fit_dict(fit_dict, number_scale, func_scale, func_registry)


def rescale_2d_fit(
    fit_dict: dict,
    xscale: float = 1.0,
    yscale: float = 1.0,
    zscale: float = 1.0,
) -> dict:
    """Rescale a 2d multi-function fit dictionary including the fit parameters,
    analysis parameters and data sums.

    Args:
        fit_dict: 2d multi-function fit dictionary.
        xscale: scalar to rescale x parameters by
        yscale: scalar to rescale y parameters by
        zscale: scalar to rescale z parameters by
    Returns:
        rescale_fit_dict: rescaled fit dictionary
    """
    fit_dict = copy.deepcopy(fit_dict)
    func_registry = FUNCTIONS2D
    num_scale = get_number_scale(xscale, yscale, zscale)
    func_scales = [num_scale, xscale, yscale, zscale]
    return rescale_fit_dict(fit_dict, num_scale, func_scales, func_registry)


class RescaleFitDicts(IterateFitDict):
    """Class to rescale any fit dictionary including multi-level fits, 1d
    multi-function fits, 2d multi-function fits and 2d sum fits. It can
    automatically determine the type of fit dictionary if not specified.
    """

    def __init__(self):
        """Inherits from the base class IterateFitDict"""
        super().__init__()

    def process_fitdict1d(
        self,
        fit_dict: dict,
        axis: Optional[str] = None,
        axis_scale: Optional[str] = None,
        xscale: float = 1.0,
        yscale: float = 1.0,
        zscale: float = 1.0,
        *args,
        **kwargs
    ) -> dict:
        """Rescale a 1d multi-function fit dictionary including the fit. Simply
        calls the rescale_1d_fit function and places it in the object for
        broader use.
        """
        return rescale_1d_fit(fit_dict, axis, axis_scale, xscale, yscale, zscale)

    def process_fitdict2d(
        self,
        fit_dict: dict,
        xscale: float = 1.0,
        yscale: float = 1.0,
        zscale: float = 1.0,
        *args,
        **kwargs
    ) -> dict:
        """Rescale a 2d multi-function fit dictionary including the fit. Simply
        calls the rescale_2d_fit function and places it in the object for
        broader use.
        """
        return rescale_2d_fit(fit_dict, xscale, yscale, zscale)

    def rescale_fitdict(
        self,
        fit_dict: dict,
        dict_type: Optional[str] = None,
        xscale: float = 1.0,
        yscale: float = 1.0,
        zscale: float = 1.0,
    ) -> dict:
        """Rescale any returned fit dictionary including multi-level fits,
        1d multi-function fits, 2d multi-function fits and 2d sum fits.
        It can automatically determine the type of fit dictionary if not
        specified. It also scales the fit parameters, analysis parameters
        and data sums.

        Args:
            fit_dict: dictionary of multi-function fit dictionaries
            dict_type: type of fit dictionary
            xscale: scalar to rescale x parameters by
            yscale: scalar to rescale y parameters by
            zscale: scalar to rescale z parameters by

        Returns:
            rescale_fit_dicts: rescaled fit dictionaries
        """
        if dict_type is None:
            dict_type = type_fitdict(fit_dict)
        rescale_func = self.fit_type_dict[dict_type]
        return rescale_func(fit_dict, xscale=xscale, yscale=yscale, zscale=zscale)


def rescale_fitdict(
    fit_dicts: dict,
    dict_type: Optional[str] = None,
    xscale: float = 1.0,
    yscale: float = 1.0,
    zscale: float = 1.0,
) -> dict:
    """
    Rescale a multi-function fit dictionary including the fit parameters,
    analysis parameters and data sums.

    Args:
        fit_dicts: dictionary of multi-function fit dictionaries
        dict_type: type of fit dictionary
        xscale: scalar to rescale x parameters by
        yscale: scalar to rescale y parameters by
        zscale: scalar to rescale z parameters by

    Returns:
        rescale_fit_dicts: rescaled fit dictionaries
    """
    rs = RescaleFitDicts()
    return rs.rescale_fitdict(fit_dicts, dict_type, xscale, yscale, zscale)
