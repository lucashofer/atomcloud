from typing import Optional, Union

import pandas as pd

from atomcloud.process_fits import type_fitdict
from atomcloud.process_fits.analyze import analyze_fitdict, AnalyzeFitDicts
from atomcloud.process_fits.format import FitDictRowFormat, row_fitdict
from atomcloud.process_fits.rescale import rescale_fitdict, RescaleFitDicts


# __all__ = [
#     "process_fitdict",
#     "process_all_fitdicts",
#     "level_fits_df",
#     "combine_level_fits",
#     "total_fit_df",
# ]


def process_fitdict(
    fit_dicts: dict,
    scale_dict: Optional[dict] = None,
    dict_type: Optional[str] = None,
    analyze: bool = True,
    row_format: bool = False,
    total_flatten: bool = False,
) -> dict:
    """
    Process the fit dictionaries for a single image fit. This can be a single
    multi-function fit or a mixes level fit. This function will determine the
    type of fit and process it accordingly.

    Args:
        fit_dicts: The fit dictionary to process
        scale_dict: A dictionary of scaling parameters to use for rescaling the
            fit dictionary. If None, no rescaling will be performed.
        dict_type: The type of fit dictionary to process. If None, the type
            will be determined automatically.
        analyze: If True, the fit dictionary will be analyzed and the
            analysis results will be added to the dictionary.
        row_format: If True, the fit dictionary will be converted to a row
            format.
        total_flatten: If True, and row_format is True and the fit dictionary
            corresponds to a mixed level fit, the mixed level multi-function
            fits will be completely flattened into a single row.
    Returns:
        The processed fit dictionary.

    """
    if dict_type is None:
        dict_type = type_fitdict(fit_dicts)
    if scale_dict is not None:
        fit_dicts = rescale_fitdict(fit_dicts, dict_type, **scale_dict)
    if analyze:
        fit_dicts = analyze_fitdict(fit_dicts, dict_type)
    if row_format:
        fit_dicts = row_fitdict(fit_dicts, dict_type, total_flatten)
    return fit_dicts


def create_dict(fit_dicts: Union[dict[dict], list[dict]]) -> dict[dict]:
    """When we are dealing with fits from multiple images, we need to make sure
    that these fit dictionaries are themselves in a dictionary where each key
    corresponds to the run image and the value is the fit dictionary. This
    function will check to see if the fit dictionaries are already in this
    format. If the fit dictionaries are in a list rather than a dictionary, this
    function will create a dictionary where the keys are simply the index of the
    fit dictionary in the list.

    Args:
        fit_dicts: The fit dictionaries to check.
    Returns:
        The fit dictionaries in a dictionary format.
    """

    if not isinstance(fit_dicts, dict):
        if isinstance(fit_dicts, list):
            indices = list(range(len(fit_dicts)))
            fit_dicts = dict(zip(indices, fit_dicts))
        else:
            raise TypeError("fit_dicts must be a list or dict")
    return fit_dicts


def process_all_fitdicts(
    fit_dicts: dict,
    scale_dict: Optional[dict] = None,
    dict_type: Optional[str] = None,
    analyze: bool = True,
    row_format: bool = False,
    total_flatten: bool = False,
) -> dict[dict]:
    """This is like process_fitdict, but it will process multiple image
    fit dictionaries at once. This is beneficial because certain processing
    objects only have to be instantiated once. This function will determine
    the type of fit dictionary and process it accordingly.

    Args:
        fit_dicts: Dictionary of image fit dictionaries to process.
    Returns
        Dictionary of processed fit dictionaries.

    See the documentation for process_fitdict for more information on the
    other input arguments as they are the same.
    """
    fit_dicts = create_dict(fit_dicts)
    if scale_dict is not None:
        sfd = RescaleFitDicts()
    if analyze:
        afd = AnalyzeFitDicts()
    if row_format:
        fdrw = FitDictRowFormat()

    processed_dicts = {}
    for run, fit_dict in fit_dicts.items():
        if dict_type is None:
            dict_type = type_fitdict(fit_dict)
        if scale_dict is not None:
            fit_dict = sfd.rescale_fitdict(fit_dict, dict_type, **scale_dict)
        if analyze:
            fit_dict = afd.analyze_fitdict(fit_dict, dict_type)
        if row_format:
            fit_dict = fdrw.row_fitdict(fit_dict, dict_type, total_flatten)
        processed_dicts[run] = fit_dict
    return processed_dicts


def combine_level_fits(flat_dicts: dict, df: bool = False) -> Union[dict, pd.DataFrame]:
    """If the fits are mixed level and only flattened for each multi-function
    fit and not entirely flattened then this function will combine each of
    the fit levels for multiple mixed level fits into a dictionary of
    dictionaries. However, each sub-dictionary will contain the fit results
    for all images at the fit level.

    Args:
        flat_dicts: The dictionary of image fit dictionaries to combine
         with the multi-function fits for each image flatted only at the
         level and not overall for each image.
        df: If True, the output will be a pandas DataFrame.

    Returns:
        The combined fit dictionaries as a dictionary of dictionaries or
        a pandas DataFrame.
    """

    combined_level_dicts = None
    flat_dicts = create_dict(flat_dicts)
    for img_key, fit_dict in flat_dicts.items():
        if combined_level_dicts is None:
            combined_level_dicts = {key: {} for key in fit_dict.keys()}
        for level_key, row in fit_dict.items():
            combined_level_dicts[level_key][img_key] = row
    if df:
        return level_fits_df(combined_level_dicts)
    else:
        return combined_level_dicts


def level_fits_df(fit_dicts: dict) -> dict[pd.DataFrame]:
    """Convert the fit dictionaries for each multi function fit
    level in a mixed level fit (see function above for more details)
    into pandas DataFrames.

    Args:
        fit_dicts: The dictionary of fit dictionaries to convert to
        pandas DataFrames.

    Returns:
        Dictionary of pandas DataFrames where each key corresponds to
        a multi-function fit level and each row in the DataFrame corresponds
        to an image.
    """

    level_dfs = {}
    for key, level_dicts in fit_dicts.items():
        level_dfs[key] = pd.DataFrame(level_dicts).transpose()
    return level_dfs


def total_fit_df(fit_dict: dict[dict]) -> pd.DataFrame:
    """Convert a dictionary of totally flattened fit dictionaries into a
    pandas DataFrame where each totallaly flattened fit dictionary is a row
    and corresponds to a different image.

    Args:
        fit_dict: The dictionary of totally flattened image fit dictionaries.
    Returns:
        The pandas DataFrame of the flattened image fit dictionaries.
    """
    return pd.DataFrame(fit_dict).transpose()
