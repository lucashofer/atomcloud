import pathlib
from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from atomcloud.utils.uncertain_utils import nominal_list


# __all__ = ["PlotBase"]


class PlotBase:
    def get_data(self, data, mask, fit_dicts, verbose=True):
        """Get the fit info and processed data for plotting.

        Args:
            data: The original data which was fit using the multi-function to
                determine the parameters (params).
            mask: The mask to be applied to the data.
            fit_dicts: Dictionary containing the fit results for a
                multi-function fit.
            verbose: If True, the data for each function will be returned.
        """

        data = self.mask_data(data, mask)
        func_strs = fit_dicts["equations"]
        constraints = fit_dicts["constraints"]
        params = nominal_list(fit_dicts["params"])

        # don't return data for each function as it's redundant for one function
        if len(func_strs) == 1:
            verbose = False
        return data, func_strs, constraints, params, verbose

    def save_plot(self, path: Union[str, Path], img_name: str) -> None:
        """
        Save the current plot to a file.

        Args:
            path: The path to the directory where the image will be saved.
            img_name: The name of the image file.

            Returns:
                None
        """
        if isinstance(path, str):
            path = Path(path)
        if isinstance(path, pathlib.WindowsPath) or isinstance(path, pathlib.PosixPath):
            plt.savefig(path / img_name, dpi=300, bbox_inches="tight")
        else:
            raise TypeError("Path must be a string or pathlib object")

    def title_string(
        self, constraints: Optional[Iterable[str]] = None, init_str: str = ""
    ) -> str:
        """
        Create a title string for a plot.

        Args:
            constraints: A list of constraints which will be applied to the
                the functions in the multi-function
                (see ConstrainedMultiFunction
                for more details)
            init_str: The initial string which will be used in the title.
                This is useful if you want to add a title to a plot which
                already has a title.

        Returns:
            The title string
        """

        title_string = init_str
        if constraints:  # if not empty
            constraint_string = "Constraints: " + " ".join(constraints)
            title_string = title_string + "\n" + constraint_string
        return title_string

    def handle_title(self, title, base_title, base_save_name):
        if title is None:
            title = ""
        fig_save_name = base_save_name + ".png"
        if title != "":
            base_title = title + " " + base_title
            fig_save_name = title + "_" + fig_save_name
        return base_title, fig_save_name

    def mask_data(
        self, data: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mask the data if a mask is provided.

        Args:
            data: The data to be masked.
            mask: The mask to be applied to the data.

        Returns:
            The masked data
        """
        if mask is not None:
            data = np.copy(data)
            data[~mask] = np.nan
        return data

    def plot_data(
        self,
        multi_func: object,
        coords: Union[np.ndarray, Iterable[np.ndarray]],
        data: np.ndarray,
        params: list[list[float]],
        func_strs: list[str],
        verbose: bool = True,
    ) -> dict:
        """
        Gets the data for the plot.

        Args:
            multi_func: The multi-function used to create the fit data.
            x: The x coordinates of the data.
            data: The original data which was fit using the multi-function to
                determine the parameters (params).
            params: The fit parameters to create the multi-function fit data.
            func_strs: The function strings used in the multi-function.
            verbose: If True, the data for each function will be returned.

        Returns:
            A dictionary containing the original data, the total fit data,
            and the fit data for each function which comprises the
            multi-function fit.
        """
        total_fit = multi_func(coords, params)
        data_dict = {"data": data, "total": total_fit}

        fdata_dict = {}
        if verbose:
            funcs = multi_func.functions
            for pars, func, fstr in zip(params, funcs, func_strs):
                fdata_dict[fstr] = func(coords, *pars)

        return {**data_dict, **fdata_dict}
