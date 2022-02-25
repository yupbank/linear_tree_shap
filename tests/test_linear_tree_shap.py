import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from linear_tree_shap import TreeExplainer
from shap import TreeExplainer as Truth

@pytest.fixture
def data():
    np.random.seed(10)
    x, y = make_regression(10000)
    return x, y

@pytest.fixture(params=range(1, 10))
def tree(data, request):
    return DecisionTreeRegressor(max_depth=request.param).fit(*data)

@pytest.fixture
def linear_treeshap(tree):
    return TreeExplainer(tree)

@pytest.fixture
def treeshap(tree):
    return Truth(tree)

def test_benchmark_linear_treeshap(data, linear_treeshap, benchmark):
    x, y = data
    benchmark(linear_treeshap.shap_values, x)

def test_benchmark_origin_treeshap(data, treeshap, benchmark):
    x, y = data
    benchmark(treeshap.shap_values, x)

def test_correctness_linear_treeshap(data, linear_treeshap, treeshap):
    x, y = data
    actual = linear_treeshap.shap_values(x)
    expected = treeshap.shap_values(x)
    np.testing.assert_array_almost_equal(actual, expected, 2)
