import copy
from typing import Optional

from atomcloud.functions import FUNCTIONS1D, FUNCTIONS2D
from atomcloud.process_fits.base import type_fitdict
from atomcloud.process_fits.iterate import IterateFitDict


# __all__ = [
#     "row_fitdict",
#     "flatten_dicts",
#     "format_1d_fit",
#     "format_2d_fit",
#     "format_sumfit",
# ]


def flatten_dicts(fit_dicts: dict) -> dict:
    """
    Flatten a dictionary of dictionaries into a single dictionary with each
    key being the key of the original dictionary plus the key of the
    nested dictionary.

    Args:
        fit_dicts: dictionary of dictionaries
    Returns:
        flattened dictionary

    """
    new_dict = {}
    for dkey, fdict in fit_dicts.items():
        for key, val in fdict.items():
            new_dict[dkey + "_" + key] = val
    return new_dict


def unique_equation_names(equations: list[str]) -> list[str]:
    """
    Get the names of the equations in a list of equations. If two equations
    are the same, append a number to the names.
    Args:
        equations: list of equations
    Returns:
        list of unique equation names

    """
    equation_names = []
    for equ in equations:
        name = equ.split("(")[0]
        equation_names.append(name)
    return equation_names


def make_equation_dict(fit_dict: dict, func_registry: object) -> dict:
    """Takes a single multi-function fit dictionary which has lists of fit
    parameters, analysis parameters, fit metrics and data sums
    and returns a dictionary where each value is a single value in the
    dictionary. Those values unique to the individual functions of the
    multi-function fit have the equaiton name appended to the key.

    Args:
        fit_dict: fit dictionary of a single multi-function fit
        func_registry: 1d or 2d function registry objects corresponding to the
            either 1d or 2d fit performed.
    Returns:
        dictionary with single values for each key
    """
    equations = fit_dict["equations"]
    equation_names = unique_equation_names(equations)
    row = {"data_sum": fit_dict["data_sum"]}

    if "fit_metrics" in fit_dict:
        for key, val in fit_dict["fit_metrics"].items():
            row[key] = val

    for index, (equ, name) in enumerate(zip(equations, equation_names)):
        func_obj = func_registry.get(equ)
        pnames = func_obj.param_dict
        for pindex, pname in enumerate(pnames):
            row[name + "_" + pname] = fit_dict["params"][index][pindex]

        if "analysis_params" in fit_dict:
            for key, val in fit_dict["analysis_params"][index].items():
                row[name + "_" + key] = val

    return row


def format_1d_fit(fit_dict: dict) -> dict:
    """Takes a single 1d fit dictionary and returns a dictionary where each
    value is a single value in the dictionary.

    Args:
        fit_dict: fit dictionary of a single 1d multi-function fit
    Returns:
        dictionary with single values for each key
    """
    fit_dict = copy.deepcopy(fit_dict)
    func_registry = FUNCTIONS1D
    return make_equation_dict(fit_dict, func_registry)


def format_2d_fit(fit_dict: dict) -> dict:
    """Takes a single 2d fit dictionary and returns a dictionary where each
    value is a single value in the dictionary.

    Args:
        fit_dict: fit dictionary of a single 2d multi-function fit
    Returns:
        dictionary with single values for each key
    """
    fit_dict = copy.deepcopy(fit_dict)
    func_registry = FUNCTIONS2D
    return make_equation_dict(fit_dict, func_registry)


def format_sumfit(fit_dict: dict) -> dict:
    """Takes a single sum fit dictionary which is composed of sub-dictionaries
    of 1d fits to the summed data on the x and y axes (along with a dictionary
    of 2d fits calculated from the 1d fits) and returns a dictionary where
    these sub-dictionaries are themselves flattened into a single dictionary
    and then these flattened dictionaries added together into a single
    dictionary.

    Args:
        fit_dict: fit dictionary of a single sum fit
    Returns:
        flatted dictionary with single values for each key
    """
    fit_dict = copy.deepcopy(fit_dict)
    func_odict = {
        "2d": format_2d_fit(fit_dict["2d"]),
        "xsum": format_1d_fit(fit_dict["xsum"]),
        "ysum": format_1d_fit(fit_dict["ysum"]),
    }
    return flatten_dicts(func_odict)


class FitDictRowFormat(IterateFitDict):
    """Format a fit dictionary into a single row dictionary for
    any type of fit dictionary.

    see IterateFitDict for more details and documentation.
    """

    def __init__(self):
        super().__init__()

    def process_fitdict1d(self, fit_dict: dict, *args, **kwargs) -> dict:
        return format_1d_fit(fit_dict)

    def process_fitdict2d(self, fit_dict: dict, *args, **kwargs) -> dict:
        return format_2d_fit(fit_dict)

    def sum_fit(self, fit_dict: dict, *args, **kwargs) -> dict:
        return format_sumfit(fit_dict)

    def row_fitdict(
        self,
        fit_dicts: dict,
        dict_type: Optional[str] = None,
        total_flatten: bool = False,
        *args,
        **kwargs
    ) -> dict:
        """Format a fit dictionary into a single row dictionary for
        any type of fit dictionary. Only the fit dict needs to be passed in
        if the fit dict type is not given, the type will be inferred from
        the fit dict.

        Lastly, if it's a mixedl level fit dictionary, the total_flatten
        argument can be used to flatten the fit dictionary into a
        single row dictionary where the levels are appended to the keys.
        Otherwise, the fit dictionary will be returned as a dictionary
        of flatted dictionaries. One flattened dictionary for each level
        (e.g. multi-function fit).

        Args:
            fit_dicts: fit dictionary to be flattened
            dict_type: type of fit dictionary. If not given, the type will
                be inferred from the fit dict.
            total_flatten: if True, the a multi-level fit dictionary will be
            completely flattened into a single row dictionary.

        Returns:
            flattened dictionary or dictionary of flattened dictionaries
            depending on the fit dictionary type and the total_flatten
            argument.
        """
        if dict_type is None:
            dict_type = type_fitdict(fit_dicts)

        func = self.fit_type_dict[dict_type]
        format_dicts = func(fit_dicts, *args, **kwargs)
        if dict_type == "mixed_level" and total_flatten is True:
            format_dicts = flatten_dicts(format_dicts)
        return format_dicts


def row_fitdict(
    fit_dicts: dict,
    dict_type: Optional[str] = None,
    total_flatten: bool = False,
    *args,
    **kwargs
) -> dict:
    """Initialize the FitDictRowFormat class and call the row_fitdict
    method. This allows any fit dict to be analyzed without having to
    initialize the class."""

    fr = FitDictRowFormat()
    return fr.row_fitdict(fit_dicts, dict_type, total_flatten, *args, **kwargs)
