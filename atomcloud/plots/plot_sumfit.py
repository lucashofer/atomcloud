# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:03:21 2022

@author: hofer
"""
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from atomcloud.functions import MultiFunction1D, MultiFunction2D
from atomcloud.plots.plot_2d import Plot2DBase
from atomcloud.utils import img_utils
from atomcloud.utils.uncertain_utils import nominal_list


# __all__ = ["Plot2DSumFit"]


class Plot2DSumFit(Plot2DBase):
    """Class for plotting 2D sum fits. Inherits from the base 2D plotting class."""

    def plot_fit(
        self,
        fit_dicts: dict,
        XY_tuple: tuple[np.ndarray, np.ndarray],
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "",
        savepath: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        *args,
        **kwargs
    ) -> None:
        """Plot the fit results from the 2D sum fit.

        Args:
            fit_dicts (dict): Dictionary containing the fit results for a
                2D multi-function fit.
            XY_tuple (tuple[np.ndarray, np.ndarray]): The x and y coordinates
                of the data.
            data (np.ndarray): The original data which was fit using the
                multi-function to determine the parameters (params).
            mask (np.ndarray, optional): The mask to be applied to the data.
            title (str, optional): The title of the plot. Defaults to ''.
            savepath (str, optional): The path to save the plot. Defaults to None.
            verbose (bool, optional): If True, the data for each function will be
                plotted. Defaults to True.
        """

        data = self.mask_data(data, mask)
        fit_dict_params = self.unpack_fit_dicts(fit_dicts)
        func_strs, constraints, params2d, xparams, yparams = fit_dict_params

        if len(func_strs) == 1:
            verbose = False

        multi_func1d = MultiFunction1D(func_strs)
        multi_func2d = MultiFunction2D(func_strs)

        x, y, xsum, ysum = img_utils.img_data_to_sums(XY_tuple, data)
        pdata_dict = self.plot_data(
            multi_func2d, XY_tuple, data, params2d, func_strs, verbose
        )
        xdata_dict = self.plot_data(multi_func1d, x, xsum, xparams, func_strs, verbose)
        ydata_dict = self.plot_data(multi_func1d, y, ysum, yparams, func_strs, verbose)

        fig, axes_2d, axes_1d, cax = self.fig_axes(func_strs, verbose)
        self.plot_2d(XY_tuple, pdata_dict, axes_2d, cax)
        self.plot_1d(x, y, xdata_dict, ydata_dict, axes_1d, "sum")

        title_base, save_name = self.handle_title(
            title, base_title="2D Sum Fit", base_save_name="sumfit"
        )
        title_str = self.title_string(constraints, title_base)
        fig.suptitle(title_str)
        if savepath is not None:
            self.save_plot(savepath, save_name)
        plt.show()

    def unpack_fit_dicts(
        self, fit_dicts: dict
    ) -> tuple[
        list[str], list[str], list[list[float]], list[list[float]], list[list[float]]
    ]:
        """Unpack the sumfit fit dictionaries into the necessary components
        for plotting as they are nested in the fit_dicts and comprised of
        the 1d and 2d fit dictionaries.

        Args:
            fit_dicts: The fit dictionaries for the 2D sum fit.

        Returns:
            func_strs: The function strings for the 2D fit.
            constraints: The constraints for the 2D fit.
            params2d: The seed parameters for the 2D fit.
            xparams: The parameters for the x-axis 1D fit.
            yparams: The parameters for the y-axis 1D fit.
        """
        func_strs = fit_dicts["2d"]["equations"]
        constraints = fit_dicts["2d"]["constraints"]
        params2d = nominal_list(fit_dicts["2d"]["params"])
        xparams = nominal_list(fit_dicts["xsum"]["params"])
        yparams = nominal_list(fit_dicts["ysum"]["params"])
        return func_strs, constraints, params2d, xparams, yparams
