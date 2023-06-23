# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:03:21 2022

@author: hofer
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from atomcloud.functions import MultiFunction1D
from atomcloud.plots.plot_base import PlotBase


# __all__ = ["Plot1DFit"]


class Plot1DFit(PlotBase):
    """Class for plotting 1D fits. Inherits from the base plotting class."""

    def plot_fit(
        self,
        fit_dicts: dict,
        x: np.ndarray,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "",
        savepath: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        *args,
        **kwargs
    ) -> None:
        """Plot the fit results

        Args:
            fit_dicts (dict): Dictionary containing the fit results for a
                1D multi-function fit.
            x (np.ndarray): The x coordinates of the data.
            data (np.ndarray): The original data which was fit using the
                multi-function to determine the parameters (params).
            mask (np.ndarray, optional): The mask to be applied to the data.
            title (str, optional): The title of the plot. Defaults to ''.
            savepath (str, optional): The path to save the plot. Defaults to None.
            verbose (bool, optional): If True, the data for each function will be
                returned. Defaults to True.
        """
        fit_info_data = self.get_data(data, mask, fit_dicts, verbose)
        data, func_strs, constraints, params, verbose = fit_info_data

        multi_func = MultiFunction1D(func_strs)
        pdata_dict = self.plot_data(multi_func, x, data, params, func_strs, verbose)
        title_base, save_name = self.handle_title(
            title, base_title="1D Fit", base_save_name="1dfit"
        )
        title_str = self.title_string(constraints, title_base)
        self.plot_1Dfit(x, pdata_dict, title_str, savepath)

    def plot_1Dfit(self, x, data_dict, title_str, savepath=None):
        """Handles the actual plotting of the 1D fit data using
        matplotlib.pyplot (see plot_fit for variable descriptions)."""

        plt.figure()
        for key, data in data_dict.items():
            plt.plot(x, data, label=key)
        plt.legend()
        plt.title(title_str)
        if savepath is not None:
            self.save_plot(savepath, title_str + "_1Dfit.png")
        plt.show()
