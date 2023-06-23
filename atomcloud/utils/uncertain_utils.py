import copy

from uncertainties import nominal_value
from uncertainties.unumpy import cos, sin


# __all__ = ['get_lab_widths', 'nominal_list']


def get_lab_widths(px, py, theta):
    """Given two orthogonal parameters which are rotated by theta with
    respect to the x and y axes, this function returns the widths in the
    lab frame.

    Args:
        px: x width
        py: y width
        theta: angle in radians

    Returns:
        list: x and y widths in the lab frame
    """
    pxx = ((px * cos(theta)) ** 2 + (py * sin(theta)) ** 2) ** 0.5
    pyy = ((px * sin(theta)) ** 2 + (py * cos(theta)) ** 2) ** 0.5
    return [pxx, pyy]


def nominal_list(param_list):
    if isinstance(param_list, list):
        param_list = copy.deepcopy(param_list)
        recursively_convert_nominal(param_list)
        return param_list
    else:
        raise TypeError("param_list must be a list")


def recursively_convert_nominal(param_list):
    for ind, param in enumerate(param_list):
        if isinstance(param, list):
            param_list[ind] = recursively_convert_nominal(param)
        else:
            param_list[ind] = nominal_value(param)
    return param_list
