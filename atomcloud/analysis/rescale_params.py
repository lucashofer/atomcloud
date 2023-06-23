from typing import Optional


# from atomcloud.uncertainty import rescale_major_minor

# __all__ = ["rescale_params", "rescale_1d_params", "rescale_2d_params"]


def rescale_parameters(
    params: list[float], indices: list[int], scale: float
) -> list[float]:
    """
    Rescale the parameters in params at the indices by the scale.

    Args:
        params: list of function parameters
        indices: list of indices which are scaled by scale
        scale: scale to rescale parameters by

    Returns:
        list of rescaled parameters
    """
    for index in indices:
        params[index] = params[index] * scale
    return params


def rescale_1d_params(
    params: list[float],
    z_indices: list[int],
    x_indices: list[int],
    xscale: float,
    zscale: float,
) -> list[float]:
    """
    Rescale fit parameters in a 1d function

    Args:
        params: list of function parameters
        zinds: list of indices which are scaled by zscale
        xinds: list of indices which are scaled by xscale
        xscale: scale to rescale x parameters by
        zscale: scale to rescale z parameters by

    Returns:
        list of rescaled parameters
    """
    # I think this copy is superfluous, but I'm not sure and don't want to break
    # anything
    scaled_params = params.copy()
    scaled_params = rescale_parameters(scaled_params, z_indices, zscale)
    scaled_params = rescale_parameters(scaled_params, x_indices, xscale)
    return scaled_params


def rescale_2d_params(
    params: list[float],
    indices: list[list],
    scales: list,
    theta_indices: Optional[list] = None,
) -> list[float]:
    """
    Rescale fit parameters in a 2d function. Must scale along x, y and z
    additionally some parameters may not lie along the x or y axis and
    so need to be rescaled differently.

    Args:
        params: list of function parameters
        indices: list of lists of indices which are scaled by the corresponding
            scale in scales
        scales: list of scales to rescale parameters by
        theta_indices: list of index of the angle parameter corresponding to
        each mixed axis tuple

    Returns:
        list of rescaled parameters
    """
    integrate_scale, xscale, yscale, zscale = scales
    x_indices, y_indices, z_indices, mixed_indices = indices
    params = rescale_parameters(params, x_indices, xscale)
    params = rescale_parameters(params, y_indices, yscale)
    params = rescale_parameters(params, z_indices, zscale)

    params = rescale_mixed_axis(params, mixed_indices, xscale, yscale, theta_indices)
    return params


def rescale_mixed_axis(
    params: list,
    mixed_inds: list[tuple[int, int]],
    xscale: float,
    yscale: float,
    theta_indices: Optional[list[int]] = None,
) -> list[float]:

    """This rescales the major and minor axes which is not super simple
    if the x and y scale are different. However, is xscale and yscale are
    the same (i.e. pixel is square) then it's pretty straight forward.

    Args:
        params: list of parameters
        mixed_inds: list of tuples of indices which are mixed axes
        xscale: scale to rescale x axis by
        yscale: scale to rescale y axis by
        theta_indices: list of index of the angle parameter corresponding to each
            tuple

    Returns:

    """
    if xscale == yscale:
        flattened_inds = [item for sublist in mixed_inds for item in sublist]
        return rescale_parameters(params, flattened_inds, xscale)
    else:
        raise NotImplementedError(
            "Rescaling mixed axes with different scales " "is not implemented yet."
        )
        # technically this codes is working, but I want to have a separate
        # uncertainty vs non uncertainty method
        # return rescale_mixed_scale(params, mixed_inds,
        #                            xscale, yscale, th_ind)


# def rescale_mixed_scale(params, mixed_inds, xscale, yscale, th_ind):
#     """Rescale the major and minor parameters which are not independent
#     but rely on both the x and y scale. This is done by finding the
#     eigenvalues and vectors and doing some fancy stuff. This isn't too hard
#     with numpy and it's in-build eigensystem decomposition. However,
#     if our parameters are uncertainty objects then we can't use numpy
#     and instead I built a custom eigenvalue decomposition method for 2x2
#     matrices only relying on uncertainties package functions. However, I need
#     to do a bit more testing on this so it's currently not implemented.

#     Additonally, our camera pixels are square so we can just rescale the
#     major and minor axes by the same amount. This is much easier and is
#     implemented in the rescale_mixed_axis function.
#     """
#     theta = params[th_ind]
#     scaled_thetas = []
#     for (xind, yind) in mixed_inds:
#         sparams = rescale_major_minor(params[xind], params[yind],
#                                       theta, xscale, yscale)
#         params[xind], params[yind], s_theta, _ = sparams
#         scaled_thetas.append(s_theta)
#     params[th_ind] = np.sum(scaled_thetas) / len(scaled_thetas)
#     return params
