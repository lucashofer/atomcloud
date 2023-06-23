from typing import Iterable, Optional, Union

import numpy as np

from atomcloud.plots import Plot1DFit, Plot2DFit, Plot2DSumFit
from atomcloud.process_fits.base import type_fitdict
from atomcloud.process_fits.iterate import IterateFitDict
from atomcloud.utils import check_1d_array, check_2d_coords


# TODO: add bool list for mixed_level_dict so that only certain levels are plotted


def check_data(data, dict_type):
    # TODO: check data for default data and coords
    if dict_type == "1dfit":
        check_1d_array(data)
    elif dict_type == "2dfit" or dict_type == "sum_fit":
        check_2d_coords(data)
    elif dict_type == "mixed_level":
        if isinstance(data, dict):
            for key, d in data.items():
                if not (isinstance(d, np.ndarray) or d is None):
                    raise TypeError("Data should be a numpy array")
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    "Data should be a numpy array or dict of \
                                    numpy arrays"
                    or None
                )


def check_mask(mask, data, dict_type):
    # TODO: check data and mask the same size
    if mask is not None:
        if dict_type == "1dfit":
            check_1d_array(mask)
            if mask.shape != data.shape:
                raise ValueError("Mask should have the same shape as data")
        elif dict_type == "2dfit" or dict_type == "sum_fit":
            check_2d_coords(mask)
            if mask.shape != data.shape:
                raise ValueError("Mask should have the same shape as data")
        elif dict_type == "mixed_level":
            if isinstance(mask, dict):
                for key, d in mask.items():
                    if isinstance(d, np.ndarray) or d is None:
                        if d is not None:
                            pass
                            # if d.shape != data[key].shape:
                            #     raise ValueError('Mask should have the same \
                            #                     shape as data')
                    else:
                        raise TypeError(
                            "Mask should be a numpy array \
                                        or None"
                        )
            else:
                if not isinstance(mask, np.ndarray):
                    raise TypeError(
                        "Data should be a numpy array or dict of \
                                    numpy arrays or None"
                    )
        # if mask.shape != data.shape:
        #     raise ValueError('Mask should have the same shape as data')


def check_mixed_level_coords(coords):
    if not (
        isinstance(coords, Iterable) or isinstance(coords, np.ndarray) or coords is None
    ):
        raise TypeError(
            "Coordinates should be a list/tuple \
                        of numpy arrays or a numpy array, or a dict of \
                        these"
        )


def check_coords(coords, dict_type):
    # TODO: check coords same size as data
    if dict_type == "1dfit":
        check_1d_array(coords)
    elif dict_type == "2dfit" or dict_type == "sum_fit":
        check_2d_coords(coords)
    elif dict_type == "mixed_level":
        if isinstance(coords, dict):
            for key, coord in coords.items():
                check_mixed_level_coords(coord)
        else:
            check_mixed_level_coords(coords)


class CloudFitPlots(IterateFitDict):
    """Class to plot fits from fit dictionaries."""

    def __init__(self):
        """The fitting objects are
        defined separately in atomcloud.plots. Here we instantiate all three
        plotting objects and call the appropriate plotting function depending
        on the type of fit dictionary."""
        super().__init__()
        self.pc1d = Plot1DFit()
        self.pc2d = Plot2DFit()
        self.pcs = Plot2DSumFit()

    def process_fitdict1d(self, fit_dicts, *args, **kwargs):
        """Plot the results from a single 1d fit dictionary."""
        fit_dicts = super().process_fitdict1d(fit_dicts)
        self.pc1d.plot_fit(fit_dicts, *args, **kwargs)

    def process_fitdict2d(self, fit_dicts, *args, **kwargs):
        """Plot the results from a single 2d fit dictionary."""
        fit_dicts = super().process_fitdict2d(fit_dicts)
        self.pc2d.plot_fit(fit_dicts, *args, **kwargs)

    def sum_fit(self, fit_dict, *args, **kwargs):
        """Plot the results from a single sum fit dictionary."""
        self.pcs.plot_fit(fit_dict, *args, **kwargs)

    def multi_level_data(self, key, data):
        if isinstance(data, dict):
            fdata = data[key]
            if fdata is None:
                fdata = data["default"]
        else:
            fdata = data
        return fdata

    def mixed_level_fit(
        self, all_fit_dicts, coords, data, mask, title, *args, **kwargs
    ):

        for key, fit_dicts in all_fit_dicts.items():
            if isinstance(mask, dict):
                fmask = mask[key]
            else:
                fmask = mask

            fdata = self.multi_level_data(key, data)
            fcoords = self.multi_level_data(key, coords)
            title = f"{title} {key}"
            self.single_level(
                fit_dicts, fcoords, fdata, fmask, title=title, *args, **kwargs
            )

    def plot_fitdict(
        self,
        fit_dicts: dict,
        coords: Union[np.ndarray, dict[str, Union[np.ndarray, Iterable[np.ndarray]]]],
        data: Union[np.ndarray, dict[str, np.ndarray]],
        dict_type: Optional[str] = None,
        mask: Union[np.ndarray, dict[str, np.ndarray]] = None,
        title: str = "",
        *args,
        **kwargs,
    ):

        if dict_type is None:
            dict_type = type_fitdict(fit_dicts)

        if not isinstance(title, str):
            raise TypeError("Title should be a string")

        check_coords(coords, dict_type)
        # check_data(data, dict_type)
        check_mask(mask, data, dict_type)

        plot_func = self.fit_type_dict[dict_type]
        plot_func(fit_dicts, coords, data, mask, title=title, *args, **kwargs)


def plot_fitdict(
    fit_dicts: dict,
    coords: Union[np.ndarray, dict[str, Union[np.ndarray, Iterable[np.ndarray]]]],
    data: Union[np.ndarray, dict[str, np.ndarray]],
    mask: Optional[Union[np.ndarray, dict[str, np.ndarray]]] = None,
    title: str = "",
    dict_type: Optional[str] = None,
    *args,
    **kwargs,
):

    cfp = CloudFitPlots()
    cfp.plot_fitdict(
        fit_dicts, coords, data, dict_type, mask, title=title, *args, **kwargs
    )
