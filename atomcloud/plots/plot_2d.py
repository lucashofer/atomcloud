# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:03:21 2022

@author: hofer
"""
import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colorbar import Colorbar

from atomcloud.functions import MultiFunction2D
from atomcloud.plots.plot_base import PlotBase
from atomcloud.utils import img_utils


# __all__ = ["Plot2DBase", "Plot2DFit"]


class Plot2DBase(PlotBase):
    def fig_axes(
        self, func_strs: list[str], verbose: bool, ax_size: float = 4.5
    ) -> tuple[plt.Figure, dict, dict, plt.Axes]:
        """
        Generates the figure and axes for the 2D fit plot.

        Args:
            func_strs: The list of function strings used in the fit.
            verbose: If True, the data for each function in the multi-function
                fit will be plotted.
            ax_size: The size of the each axis.

        Returns:
            A tuple containing the figure, a dictionary of the 2D axes, a
            dictionary of the 1D axes, and the colorbar axis. The keys for the
            2D axes are 'data', 'total', and the function strings. The keys for
            the 1D axes are 'x' and 'y'.

        """
        row_num = 2
        col_num = 2
        num_funcs = len(func_strs)
        if verbose:
            func_col_num = int(math.ceil(num_funcs / 2))
            col_num = col_num + func_col_num

        fig = plt.figure(
            figsize=((col_num + 1 / 4) * ax_size, row_num * ax_size)
        )  # generates the figure size
        width_ratios = [15 for _ in range(col_num)] + [1]
        gs = gridspec.GridSpec(
            row_num, col_num + 1, height_ratios=[1, 1], width_ratios=width_ratios
        )  # generates the subplots
        gs.update(hspace=0.2, wspace=0.2)
        # axes = self.get_data_axes(fig, gs, row_num, col_num)
        axes = [
            [fig.add_subplot(gs[i, j]) for j in range(col_num)] for i in range(row_num)
        ]

        cax = plt.subplot(gs[:, -1])
        self.set_colorbar_position(axes, cax)

        # set all plot axes as squares
        [ax.set_box_aspect(1) for col_axes in axes for ax in col_axes]
        # get axes for different fitting sections
        axes_2d = {"data": axes[0][0], "total": axes[1][1]}
        axes_1d = {"x": axes[1][0], "y": axes[0][1]}
        # get individual function imshow axes
        if verbose:
            func_axes = [
                ax
                for i in range(func_col_num)
                for ax in [axes[0][2 + i], axes[1][2 + i]]
            ]

            if func_col_num * row_num > num_funcs:
                func_axes[-1].axis("off")

            func_axes_dict = dict(zip(func_strs, func_axes))
            axes_2d.update(func_axes_dict)

        return fig, axes_2d, axes_1d, cax

    def plot_2d(
        self,
        coords: tuple[np.ndarray, np.ndarray],
        data_dict: dict[str, np.ndarray],
        axes_2d: dict[str, plt.Axes],
        cax: plt.Axes,
    ) -> None:
        """Plots the 2D fit data including the original data, the total fit
        data, and the individual function fit data both for the sum data and
        the full 2d data.

        Args:
            coords: The tuple containing the x and y coordinates of the data.
            data_dict: The dictionary containing the 2d data, total 2d fit data
                and the individual function 2d fit data.
            axes_2d: The dictionary containing the 2d axes in the figure.
            cax: The colorbar axis.
        """
        if len(data_dict) <= 2:
            vkeys = list(data_dict.keys())
        else:
            vkeys = ["data", "total"]

        vdata = np.stack([data_dict[key] for key in vkeys])
        vmin, vmax = np.nanmin(vdata), np.nanmax(vdata)
        for key in data_dict:
            axes_2d[key].pcolormesh(
                *coords, data_dict[key], shading="auto", vmin=vmin, vmax=vmax
            )
            axes_2d[key].set_title(key)
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        Colorbar(
            ax=cax,
            mappable=cm.ScalarMappable(norm=cNorm),
            orientation="vertical",
            ticklocation="right",
        )

    def set_colorbar_position(self, axes: list[list[plt.Axes]], cax: plt.Axes):
        """Set the position of the colorbar axis.

        Args:
            axes: The list of data axes in the figure.
            cax: The colorbar axis.
        """
        bbox = cax.get_position()
        bbox_width = bbox.x1 - bbox.x0
        bbox.x0 = axes[-1][-1].get_position().x1 + 0.01
        bbox.x1 = bbox.x0 + bbox_width
        cax.set_position(bbox)

    def plot_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xdata_dict: dict[str, np.ndarray],
        ydata_dict: dict[str, np.ndarray],
        ax_dict: dict[str, plt.Axes],
        ptype: str,
    ):
        """
        Plots the 1D sum data for the x and y axes.

        Args:
            x: The x coordinates of the data.
            y: The y coordinates of the data.
            xdata_dict: The dictionary containing the x-axis data.
            ydata_dict: The dictionary containing the y-axis data.
            ax_dict: The dictionary containing the x and y axes.
        """
        for key in xdata_dict:
            ax_dict["x"].plot(x, xdata_dict[key], label=key)
            ax_dict["y"].plot(ydata_dict[key], y, label=key)
        ax_dict["x"].set_title("X-Axis " + ptype)
        ax_dict["y"].set_title("Y-Axis " + ptype)
        [ax.legend() for _, ax in ax_dict.items()]


class Plot2DFit(Plot2DBase):
    """make a more general plotting class that can be used in a variety of objects"""

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
    ):

        fit_info_data = self.get_data(data, mask, fit_dicts, verbose)
        data, func_strs, constraints, params, verbose = fit_info_data

        multi_func = MultiFunction2D(func_strs)
        pdata_dict = self.plot_data(
            multi_func, XY_tuple, data, params, func_strs, verbose
        )
        x, y, xdata_dict, ydata_dict = self.get_data1d(XY_tuple, pdata_dict, mask)

        fig, axes_2d, axes_1d, cax = self.fig_axes(func_strs, verbose)
        self.plot_2d(XY_tuple, pdata_dict, axes_2d, cax)
        self.plot_1d(x, y, xdata_dict, ydata_dict, axes_1d, "Sum")

        title_base, save_name = self.handle_title(
            title, base_title="2D Fit", base_save_name="2dfit"
        )
        title_str = self.title_string(constraints, title_base)
        fig.suptitle(title_str)
        if savepath is not None:
            self.save_plot(savepath, save_name)
        plt.show()

    def sum_data_mask(self, XY_tuple, data, mask):
        """Sum the data in the masked region, but particularly for the case
        of weird masks this is a hacky way to do this, but it works for now"""

        sum_data = np.copy(data)
        X, Y = XY_tuple
        xmin, xmax = np.amin(X), np.amax(X)
        ymin, ymax = np.amin(Y), np.amax(Y)
        if mask is not None:
            mxmin, mxmax = np.amin(X[mask]), np.amax(X[mask])
            mymin, mymax = np.amin(Y[mask]), np.amax(Y[mask])
            if xmin == mxmin and xmax == mxmax and ymin == mymin and ymax == mymax:
                xmask = (X >= mxmin) & (X <= mxmax)
                ymask = (Y >= mymin) & (Y <= mymax)
                mask = xmask & ymask
            sum_data[~mask] = 0
        return sum_data

    def get_data1d(
        self,
        XY_tuple: tuple[np.ndarray, np.ndarray],
        data_dict: dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Get the 1D data for the x and y axes for the sum data, total fit
        sum data, and the individual function sum data.

        Args:
            XY_tuple: The tuple containing the x and y coordinates of the data.
            data_dict: The dictionary containing the 2d data, total 2d fit data
            and the individual function 2d fit data.
            mask: The mask to be applied to the data.

        Returns:
            A tuple containing the x and y coordinates of the data, and the
            x and y data dictionaries for the sum data, total fit sum data,
            and the individual function sum data.
        """
        x, y = img_utils.tuple_coords1d(XY_tuple)
        xdata_dict = {}
        ydata_dict = {}
        for key, data in data_dict.items():
            if key == "data" and mask is not None:
                data = self.sum_data_mask(XY_tuple, data, mask)
            xsum, ysum = img_utils.sum_img_data(data)
            xdata_dict[key] = xsum
            ydata_dict[key] = ysum
        return x, y, xdata_dict, ydata_dict
