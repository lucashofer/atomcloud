from typing import Optional

import numpy as np


# __all__ = ["type_fitdict", "get_level_type"]


def get_level_type(fit_dicts: dict, dict_type: Optional[str] = None) -> str:
    """Takes a 1d multi-function, 2d multi-function, or sum fit fit dictionary
    and determines the type of fit dictionary it is.

    Args:
        fit_dicts: fit dictionary
        dict_type: optional, the type of fit dictionary

    Returns:
        The type of fit dictionary
    """

    dict_keys = fit_dicts.keys()
    if "fit_type" in dict_keys:
        cloud_fit_type = fit_dicts["fit_type"]
        if cloud_fit_type == "1d":
            dict_type = "1dfit"
        elif cloud_fit_type == "2d":
            dict_type = "2dfit"
        else:
            raise TypeError("I Dont recognize this cloudfit dict type")
    else:
        sum_keys = {"xsum", "ysum", "2d"}
        if set(dict_keys) == sum_keys:
            dict_type = "sum_fit"
    return dict_type


def type_fitdict(fit_dicts: dict) -> dict:
    """Takes a multi-level, 1d multi-function, 2d multi-function, or sum fit
    fit dictionary and determines the type of fit dictionary it is.

    Args:
        fit_dicts: fit dictionary

    Returns:
        The type of fit dictionary
    """
    if isinstance(fit_dicts, dict):
        dict_type = get_level_type(fit_dicts)
        if dict_type is None:
            level_types = [get_level_type(fit_dict) for fit_dict in fit_dicts.values()]
            if np.all([dtype is not None for dtype in level_types]):
                dict_type = "mixed_level"
            else:
                raise TypeError("I Dont recognize this dictionary format")
    else:
        raise TypeError("Fit dict should be a dictionary")
    return dict_type
