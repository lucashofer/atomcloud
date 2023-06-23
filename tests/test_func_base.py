import numpy as np
import pytest

from atomcloud.functions.func_base import FunctionBase


class ConcreteFunction(FunctionBase):
    def create_function(self, backend):
        def function(coords, a, b, c):
            x, y, z = coords
            return a * x + b * y + c * z

        return function


@pytest.fixture
def test_function():
    return ConcreteFunction()


def test_call(test_function):
    coords = (1, 2, 3)
    result = test_function(coords, 1, 2, 3)
    assert result == 14


def test_make_function(test_function):
    np_func = test_function.make_function(use_jax=False)
    coords = (1, 2, 3)
    result = np_func(coords, 1, 2, 3)
    assert result == 14


def test_initial_seed(test_function):
    coords = np.array([1, 2, 3])
    data = np.array([14])
    seed = test_function.initial_seed(coords, data)
    assert seed == [1.0, 1.0, 1.0]


def test_default_bounds(test_function):
    min_bounds, max_bounds = test_function.default_bounds()
    assert min_bounds == [-np.inf, -np.inf, -np.inf]
    assert max_bounds == [np.inf, np.inf, np.inf]


def test_create_parameter_dict(test_function):
    test_function.create_parameter_dict()
    assert test_function.param_dict == ["a", "b", "c"]


def test_analyze_parameters(test_function):
    params = [1.0, 2.0, 3.0]
    analysis_params = test_function.analyze_parameters(params)
    assert analysis_params == {}


def test_rescale_parameters(test_function):
    params = [1.0, 2.0, 3.0]
    scales = [1.0, 2.0, 3.0]
    rescaled_params = test_function.rescale_parameters(params, scales)
    assert rescaled_params == params


def test_rescale_analysis_params(test_function):
    analysis_params = {"param1": 1.0, "param2": 2.0}
    scales = [1.0, 2.0, 3.0]
    rescaled_analysis_params = test_function.rescale_analysis_params(
        analysis_params, scales
    )
    assert rescaled_analysis_params == analysis_params


# def test_make_function_exception(test_function, monkeypatch):
#     monkeypatch.setattr(test_function, 'jnp', None)
#     with pytest.raises(Exception, match='JAX is not installed'):
#         test_function.make_function(use_jax=True)


def test_call_with_different_coords_types(test_function):
    coords_tuple = (1, 2, 3)
    coords_list = [1, 2, 3]
    coords_array = np.array([1, 2, 3])

    result_tuple = test_function(coords_tuple, 1, 2, 3)
    result_list = test_function(coords_list, 1, 2, 3)
    result_array = test_function(coords_array, 1, 2, 3)

    assert result_tuple == result_list == result_array
