from atomcloud.mixed_level.mixed_base import fit_level, MixedLevelBase
from atomcloud.utils import fit_utils


class SimpleThermal(MixedLevelBase):

    fit_order = ["gsum", "thermal"]

    @fit_level("SumFit2D", ["gaussian", "foffset"])
    def gsum(self, coords, data, seed=None, mask=None, verbose=False):
        return coords, data, seed, mask

    @fit_level("2DFit", ["febose", "foffset"], ["x0"])
    def thermal(self, coords, data, seed=None, mask=None, verbose=False):
        params_2d = self.level_fits["gsum"]["2d"]["params"]

        # get elliptical mask
        gauss_params, _ = params_2d
        gauss_centroid = gauss_params[1:3]
        mask_width_angle = self.kwargs["mask_params"]
        mask_params = gauss_centroid + mask_width_angle
        if "scalar" in self.kwargs.keys():
            scalar = self.kwargs["scalar"]
        else:
            scalar = 1.0

        self.bec_mask = fit_utils.generate_elliptical_mask2D(
            coords, *mask_params, scale=scalar
        )
        thermal_mask = ~self.bec_mask  # invert mask to exclude the BEC
        return coords, data, params_2d, thermal_mask
