import numpy as np

from atomcloud.mixed_level.mixed_base import fit_level, MixedLevelBase
from atomcloud.utils import fit_utils, mask_utils


class SimpleBimodal(MixedLevelBase):

    fit_order = ["gsum", "thermal", "tf_sum", "bec"]

    @fit_level("SumFit2D", ["gaussian", "foffset"])
    def gsum(self, coords, data, seed=None, mask=None, verbose=False):
        return coords, data, seed, mask

    @fit_level("2DFit", ["febose", "foffset"])
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

    @fit_level("SumFit2D", ["parabola", "foffset"])
    def tf_sum(self, coords, data, seed=None, mask=None, verbose=False):
        thermal_params = self.level_fits["thermal"]["params"]
        fit_func = self.level_fit_objs["thermal"].func
        thermal_density = fit_func(coords, thermal_params)
        self.bec_density = data - thermal_density
        square_mask = mask_utils.find_square_mask(self.bec_mask)
        return coords, data, seed, square_mask

    @fit_level("2DFit", ["tf"])
    def bec(self, coords, data, seed=None, mask=None, verbose=False):
        params2d = self.level_fits["tf_sum"]["2d"]["params"]
        bec_param, _ = params2d
        bec_seed = [bec_param]
        return coords, self.bec_density, bec_seed, self.bec_mask


class ComplexBimodal(MixedLevelBase):

    fit_order = ["gsum", "thermal1", "thermal2", "thermal3", "tfsum", "bec"]

    def get_fit(self, coords, data, *args, **kwargs):
        self.thermal_passed = False
        fit_passed, fit_info = super().get_fit(coords, data, *args, **kwargs)
        fit_info["bec_sum"] = self.bec_sums(coords, data)
        return self.thermal_passed, fit_info

    @fit_level("SumFit2D", ["gaussian", "foffset"])
    def gsum(self, coords, data, seed=None, mask=None, verbose=False):
        return coords, data, seed, mask

    @fit_level("2DFit", ["febose", "foffset"])
    def thermal1(self, coords, data, seed=None, mask=None, verbose=False):
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

    @fit_level("2DFit", ["febose", "foffset"])
    def thermal2(self, coords, data, seed=None, mask=None, verbose=False):
        thermal1_params = self.level_fits["thermal1"]["params"]
        febose_params, off_params = thermal1_params
        emask = fit_utils.generate_elliptical_mask2D(
            coords, *febose_params[1:], scale=1.75
        )
        self.thermal_mask = ~emask  # invert mask to exclude the BEC
        ebose_params = febose_params
        # ebose_params = ebose_params + [1] uncomment for ebose rather than febose
        seed = [ebose_params, off_params]
        return coords, data, seed, self.thermal_mask

    @fit_level("2DFit", ["febose", "foffset"])
    def thermal3(self, coords, data, seed=None, mask=None, verbose=False):
        thermal1_params = self.level_fits["thermal2"]["params"]
        febose_params, off_params = thermal1_params
        inside_thermal_mask = ~self.thermal_mask  # invert mask to exclude the BEC
        outside_bec_mask = ~self.bec_mask
        total_mask = inside_thermal_mask & outside_bec_mask
        ebose_params = febose_params
        # ebose_params = ebose_params + [.8] uncomment for ebose rather than febose
        seed = [ebose_params, off_params]
        self.thermal_passed = False
        return coords, data, seed, total_mask

    @fit_level("SumFit2D", ["parabola", "foffset"])
    def tfsum(self, coords, data, seed=None, mask=None, verbose=False):
        self.thermal_passed = True
        thermal_params = self.level_fits["thermal3"]["params"]
        fit_func = self.level_fit_objs["thermal3"].func
        thermal_density = fit_func(coords, thermal_params)
        self.bec_density = data - thermal_density
        square_mask = mask_utils.find_square_mask(self.bec_mask)
        return coords, data, seed, square_mask

    @fit_level("2DFit", ["tf"])
    def bec(self, coords, data, seed=None, mask=None, verbose=False):
        params2d = self.level_fits["tfsum"]["2d"]["params"]
        bec_param, _ = params2d
        bec_seed = [bec_param]
        return coords, self.bec_density, bec_seed, self.bec_mask

    def bec_sums(self, coords, data):
        level_bec_sum = {}
        for key in self.level_fits.keys():
            if "thermal" in key:
                thermal_params = self.level_fits[key]["params"]
                fit_func = self.level_fit_objs[key].func
                thermal_density = fit_func(coords, thermal_params)
                bec_density = data - thermal_density
                level_bec_sum[key] = np.sum(bec_density[self.bec_mask])
        return level_bec_sum
