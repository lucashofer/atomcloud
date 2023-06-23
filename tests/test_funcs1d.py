import numpy as np
import pytest

from atomcloud.functions.funcs_1d import (
    EnhancedBose1D,
    FixedEnhancedBose1D,
    FixedOffset1D,
    Gaussian1D,
    Parabola1D,
    ThomasFermi1D,
)
from atomcloud.utils import fit_utils


@pytest.fixture
def test_gaussian1d():
    return Gaussian1D()


@pytest.fixture
def test_parabola1d():
    return Parabola1D()


@pytest.fixture
def test_thomasfermi1d():
    return ThomasFermi1D()


@pytest.fixture
def test_enhancedbose1d():
    return EnhancedBose1D()


@pytest.fixture
def test_fixedenhancedbose1d():
    return FixedEnhancedBose1D()


@pytest.fixture
def test_fixedoffset1d():
    return FixedOffset1D()


# Add tests for the other child classes following the same pattern


def test_initial_seed(test_gaussian1d):
    x = np.linspace(0, 10, 11)
    data = np.exp(-((x - 5) ** 2) / 2)
    seed = test_gaussian1d.initial_seed(x, data)
    assert len(seed) == 3
    assert seed[0] == pytest.approx(np.max(data), rel=1e-2)
    assert seed[2] == pytest.approx(x[np.argmax(data)], rel=1e-2)


def test_gaussian1d_call(test_gaussian1d):
    x = np.linspace(0, 10, 11)
    params = [1, 1, 5]
    result = test_gaussian1d(x, *params)
    expected = np.exp(-((x - 5) ** 2) / 2)
    assert np.allclose(result, expected, rtol=1e-2)


def test_rescale_parameters_function1dbase(test_gaussian1d):
    params = [1.0, 2.0, 3.0]
    scales = [2.0, 3.0, 4.0]
    rescaled_params = test_gaussian1d.rescale_parameters(params, scales)
    expected_params = [4.0, 6.0, 9.0]
    assert np.allclose(rescaled_params, expected_params)


def test_analyze_parameters_function1dbase(test_gaussian1d):
    params = [1.0, 2.0, 3.0]
    analysis_params = test_gaussian1d.analyze_parameters(params)
    assert isinstance(analysis_params, dict)
    assert "int" in analysis_params


def test_rescale_analysis_params_function1dbase(test_gaussian1d):
    analysis_params = {"int": 10.0}
    scales = [2.0, 3.0, 4.0]
    rescaled_analysis_params = test_gaussian1d.rescale_analysis_params(
        analysis_params, scales
    )
    expected_params = {"int": 20.0}
    assert rescaled_analysis_params == expected_params


def test_create_function_gaussian1d(test_gaussian1d):
    function = test_gaussian1d.create_function(np)
    assert callable(function)
    assert function.__name__ == "gaussian1d"


def test_integrate_function_gaussian1d(test_gaussian1d):
    params = [1.0, 2.0, 3.0]
    integral = test_gaussian1d.integrate_function(params)
    expected_integral = np.sqrt(2 * np.pi) * 1.0 * 2.0
    assert np.isclose(integral, expected_integral)


##########################################################################


def test_create_polylog1d_fixedenhancedbose1d(test_fixedenhancedbose1d):
    polylog1d = test_fixedenhancedbose1d.create_polylog1d(np)
    assert callable(polylog1d)

    # Test the polylog1d function on sample data
    z = np.array([0.1, 0.5, 1.0])
    gamma = 2.5
    polylog_result = polylog1d(z, gamma)
    assert polylog_result.shape == z.shape


def test_create_function_fixedenhancedbose1d(test_fixedenhancedbose1d):
    thermal_cloud = test_fixedenhancedbose1d.create_function(np)
    assert callable(thermal_cloud)

    # Test the thermal_cloud function on sample data
    x = np.linspace(0, 10, 11)
    params = [1.0, 2.0, 5.0]
    result = thermal_cloud(x, *params)
    assert result.shape == x.shape


def test_integrate_function_fixedenhancedbose1d(test_fixedenhancedbose1d):
    params = [1.0, 2.0, 3.0]
    integral = test_fixedenhancedbose1d.integrate_function(params)
    poly_log = fit_utils.polylog_val(1, 3)
    expected_integral = (2 * np.pi) ** 0.5 * 1.0 * 2.0 * poly_log
    assert np.isclose(integral, expected_integral)


def test_create_polylog1d_custom_n_max_fixedenhancedbose1d(test_fixedenhancedbose1d):
    n_max = 10
    polylog1d = test_fixedenhancedbose1d.create_polylog1d(np, n_max=n_max)
    assert callable(polylog1d)

    # Test the polylog1d function on sample data
    z = np.array([0.1, 0.5, 1.0])
    gamma = 2.5
    polylog_result = polylog1d(z, gamma)
    assert polylog_result.shape == z.shape


def test_create_polylog1d_edge_case_fixedenhancedbose1d(test_fixedenhancedbose1d):
    polylog1d = test_fixedenhancedbose1d.create_polylog1d(np)
    assert callable(polylog1d)

    # Test the polylog1d function on edge case input
    z = np.zeros(3)
    gamma = 2.5
    polylog_result = polylog1d(z, gamma)
    assert np.all(polylog_result == np.zeros(3))


# def test_thermal_cloud_edge_case_fixedenhancedbose1d(test_fixedenhancedbose1d):
#     thermal_cloud = test_fixedenhancedbose1d.create_function(np)
#     assert callable(thermal_cloud)

#     # Test the thermal_cloud function on edge case input
#     x = np.zeros(3)
#     params = [1.0, 2.0, 3.0]
#     result = thermal_cloud(x, *params)
#     assert np.all(result == np.ones(3))


##########################################################################


def test_thermal_cloud_sample_input_enhancedbose1d(test_enhancedbose1d):
    thermal_cloud = test_enhancedbose1d.create_function(np)
    assert callable(thermal_cloud)

    # Test the thermal_cloud function on sample input
    x = np.array([0.0, 0.5, 1.0])
    params = [1.0, 2.0, 3.0, 0.5]
    result = thermal_cloud(x, *params)
    assert result.shape == x.shape


# def test_thermal_cloud_edge_case_enhancedbose1d(test_enhancedbose1d):
#     thermal_cloud = test_enhancedbose1d.create_function(np)
#     assert callable(thermal_cloud)

#     # Test the thermal_cloud function on edge case input
#     x = np.zeros(3)
#     params = [1.0, 2.0, 3.0, 0.5]
#     result = thermal_cloud(x, *params)
#     assert np.all(result == np.ones(3))


def test_integrate_function_sample_input_enhancedbose1d(test_enhancedbose1d):
    params = [1.0, 2.0, 3.0, 0.5]
    result = test_enhancedbose1d.integrate_function(params)
    assert isinstance(result, float)


def test_default_bounds_enhancedbose1d(test_enhancedbose1d):
    bounds = test_enhancedbose1d.default_bounds()
    assert len(bounds) == 2
    assert len(bounds[0]) == 4
    assert len(bounds[1]) == 4
    assert bounds[0] == [0, 0, -np.inf, 0]
    assert bounds[1] == [np.inf, np.inf, np.inf, 1]


##########################################################################


def test_parabola1d_sample_input_parabola1d(test_parabola1d):
    parabola1d = test_parabola1d.create_function(np)
    assert callable(parabola1d)

    # Test the parabola1d function on sample input
    x = np.array([3.0, 3.5, 4.0])
    params = [1.0, 2.0, 3.0]
    result = parabola1d(x, *params)
    expected = np.array([1.0, 0.9375, 0.75])
    assert np.allclose(result, expected)


def test_integrate_function_sample_input_parabola1d(test_parabola1d):
    params = [1.0, 2.0, 3.0]
    result = test_parabola1d.integrate_function(params)
    expected = 8 / 3
    assert np.isclose(result, expected)


def test_thomas_fermi1d_sample_input_thomas_fermi_bec(test_thomasfermi1d):
    thomas_fermi_bec = test_thomasfermi1d.create_function(np)
    assert callable(thomas_fermi_bec)

    # Test the thomas_fermi_bec function on sample input
    x = np.array([3.0, 3.5, 4.0])
    params = [1.0, 2.0, 3.0]
    result = thomas_fermi_bec(x, *params)
    expected = np.array([1.0, 0.87890625, 0.5625])
    assert np.allclose(result, expected)


def test_integrate_function_sample_input_thomas_fermi1d(test_thomasfermi1d):
    params = [1.0, 2.0, 3.0]
    result = test_thomasfermi1d.integrate_function(params)
    expected = 16 / 15 * 2.0
    assert np.isclose(result, expected)


def test_fixed_offset1d_sample_input_fixed_offset(test_fixedoffset1d):
    fixed_offset = test_fixedoffset1d.create_function(np)
    assert callable(fixed_offset)

    # Test the fixed_offset function on sample input
    x = np.array([3.0, 3.5, 4.0])
    params = [1.0]
    result = fixed_offset(x, *params)
    expected = np.array([1.0, 1.0, 1.0])
    assert np.allclose(result, expected)


def test_initial_seed_sample_input_fixed_offset1d(test_fixedoffset1d):
    x = np.array([1.0, 2.0, 3.0])
    data = np.array([1.5, 0.5, 2.0])
    result = test_fixedoffset1d.initial_seed(x, data)
    expected = [0.5]
    assert np.allclose(result, expected)


def test_rescale_parameters_sample_input_fixed_offset1d(test_fixedoffset1d):
    params = [1.0]
    scale = [2.0, 1.5, 0.5]
    result = test_fixedoffset1d.rescale_parameters(params, scale)
    expected = [0.5]
    assert np.allclose(result, expected)
