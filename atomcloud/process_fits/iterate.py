import copy
from abc import ABC, abstractmethod

from atomcloud.functions import FUNCTIONS1D, FUNCTIONS2D
from atomcloud.process_fits import base


class IterateFitDict(ABC):
    def __init__(self):
        super().__init__()
        self.func_registry1d = FUNCTIONS1D
        self.func_registry2d = FUNCTIONS2D
        self.level_fit_type = {
            "1dfit": self.process_fitdict1d,
            "2dfit": self.process_fitdict2d,
            "sum_fit": self.sum_fit,
        }

        self.fit_type_dict = self.level_fit_type.copy()
        self.fit_type_dict["mixed_level"] = self.mixed_level_fit

    @abstractmethod
    def process_fitdict1d(self, fit_dict: dict, *args, **kwargs) -> dict:
        """Process a single 1d fit dictionary. It's important to deepcopy the
        dictionary before processing it to avoid changing the original
        dictionary.
        """
        fit_dict = copy.deepcopy(fit_dict)
        return fit_dict

    @abstractmethod
    def process_fitdict2d(self, fit_dict: dict, *args, **kwargs) -> dict:
        """Process a single 2d fit dictionary. It's important to deepcopy the
        dictionary before processing it to avoid changing the original
        dictionary.
        """
        fit_dict = copy.deepcopy(fit_dict)
        return fit_dict

    def sum_fit(self, fit_dict: dict[dict], *args, **kwargs) -> dict:
        """Process a single sum fit dictionary. The sum fit dictionary is
        composed of sub-dictionaries of 1d fits to the summed data on the x and
        y axes (along with a dictionary of 2d fits calculated from the 1d fits).
        These are already deepcopied in their respective processing functions.
        """
        func_dict = {
            "2d": (self.process_fitdict2d, None),
            "xsum": (self.process_fitdict1d, "x"),
            "ysum": (self.process_fitdict1d, "y"),
        }

        func_output_dict = {}
        for key, dict_vals in func_dict.items():
            func, axis = dict_vals
            func_output_dict[key] = func(fit_dict[key], axis=axis, *args, **kwargs)
        return func_output_dict

    def mixed_level_fit(self, all_fit_dicts: dict[dict], *args, **kwargs) -> dict:
        """Process a mixed level fit dictionary. The mixed level fit is
        composed of either 1d multi-function fits, 2d multi-function fits, or
        sum fits. This function iterates through each fit level and calls the
        appropriate processing function for each fit level.

        Args:
            all_fit_dicts: dictionary of fit dictionaries
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            dictionary of processed fit dictionaries
        """
        all_level_dicts = {}
        for key, fit_dict in all_fit_dicts.items():
            all_level_dicts[key] = self.single_level(fit_dict, *args, **kwargs)
        return all_level_dicts

    def single_level(self, fit_dict: dict, *args, **kwargs) -> dict:
        """Process a single level from a multi-level fit dictionary.
        The single level fit is either a 1d multi-function fit,
        2d multi-function fit, or sum fits. This function calls the appropriate
        processing function for the fit level.

        Args:
            fit_dict: dictionary of fit dictionaries
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            dictionary of processed fit dictionaries
        """
        dict_type = base.get_level_type(fit_dict)
        return self.fit_type_dict[dict_type](fit_dict, *args, **kwargs)
