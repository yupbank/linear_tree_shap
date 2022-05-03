import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import linear_tree_shap 
import fasttreeshap
import shap
from shap import TreeExplainer as Truth
import pandas as pd
from sklearn.model_selection import train_test_split

def dummy_transform(df):
    categorical_feature_names = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]
    for name in categorical_feature_names:
        dummy_df = pd.get_dummies(df[name])
        if "?" in dummy_df.columns.values:
            dummy_df.drop("?", axis=1, inplace=True)
        df = pd.concat([df, dummy_df], axis=1)
        df.drop(name, axis=1, inplace=True)
    return df

def load_conductor():
    data_path = '/Users/pengyu/Downloads/Fast_TreeSHAP_Code/dataset/superconduct'
    data = pd.read_csv(data_path+"/train.csv", engine = "python")
    train, test = train_test_split(data, test_size = 0.5, random_state = 0)
    label_train = train["critical_temp"]
    label_test = test["critical_temp"]
    train = train.iloc[:, :-1]
    test = test.iloc[:, :-1]
    return train.values, label_train, test.values

def load_adult():
    data_path = '/Users/pengyu/Downloads/Fast_TreeSHAP_Code/dataset/adult'
    train = pd.read_csv(data_path+'/adult.data', sep = ",\s+", header = None, skiprows = 1, engine = "python")    
    test = pd.read_csv(data_path+"/adult.test", sep = ",\s+", header = None, skiprows = 1, engine = "python")
    label_train = train[14].map({"<=50K": 0, ">50K": 1}).tolist()
    label_test = test[14].map({"<=50K.": 0, ">50K.": 1}).tolist()
    train = train.iloc[:, :-2]
    test = test.iloc[:, :-2]
    feature_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                     "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"]
    train.columns = feature_names
    test.columns = feature_names
    train = dummy_transform(train)
    test = dummy_transform(test)
    return train.values, label_train, test.values



@pytest.fixture(params=[pytest.param(2, marks= [pytest.mark.depth2, pytest.mark.simple]),
                        pytest.param(4, marks= [pytest.mark.depth4, pytest.mark.simple]),
                        pytest.param(6, marks= [pytest.mark.depth6, pytest.mark.simple]),
                        pytest.param(8, marks= [pytest.mark.depth8, pytest.mark.moderate]),
                        pytest.param(10, marks=[pytest.mark.depth10, pytest.mark.moderate]),
                        pytest.param(12, marks=[pytest.mark.depth12, pytest.mark.slow]),
                        pytest.param(16, marks=[pytest.mark.depth16, pytest.mark.slow]),
                        ])
def tree(data, request):
    x, y, x_test = data
    return DecisionTreeRegressor(max_depth=request.param).fit(x, y)

@pytest.fixture
def linear_treeshap(tree):
    return linear_tree_shap.TreeExplainer(tree)

@pytest.fixture
def treeshap(tree):
    return shap.TreeExplainer(tree)

@pytest.fixture
def fast_treeshap_v2(tree):
    return fasttreeshap.TreeExplainer(tree, algorithm='v2', n_jobs=1)

@pytest.fixture
def fast_treeshap_v1(tree):
    return fasttreeshap.TreeExplainer(tree, algorithm='v1', n_jobs=1)

class TestAdult:
    @pytest.fixture
    def data(self):
        return load_adult()

    def test_benchmark_linear_treeshap_v2(self, data, linear_treeshap, benchmark):
        x, y, x_test = data
        benchmark(linear_treeshap.shap_values_v2, x_test)

    def test_benchmark_linear_treeshap(self, data, linear_treeshap, benchmark):
        x, y, x_test = data
        benchmark(linear_treeshap.shap_values, x_test)


    def test_benchmark_treeshap(self, data, treeshap, benchmark):
        x, y, x_test = data
        benchmark(treeshap.shap_values, x_test, check_additivity=False)

    def test_benchmark_fast_treeshap_v1(self, data, fast_treeshap_v1, benchmark):
        x, y, x_test = data
        benchmark(fast_treeshap_v1.shap_values, x_test, check_additivity=False)

    def test_benchmark_fast_treeshap_v2(self, data, fast_treeshap_v2, benchmark):
        x, y, x_test = data
        benchmark(fast_treeshap_v2.shap_values, x_test, check_additivity=False)

    def test_correctness_linear_treeshap(self, data, linear_treeshap, treeshap, tree):
        x, y, x_test = data
        actual = linear_treeshap.shap_values(x_test)
        expected = treeshap.shap_values(x_test)
        np.testing.assert_array_almost_equal(actual, expected, 2)

    def test_correctness_linear_treeshap_v2(self, data, linear_treeshap, treeshap, tree):
        x, y, x_test = data
        actual = linear_treeshap.shap_values_v2(x_test)
        expected = treeshap.shap_values(x_test)
        np.testing.assert_array_almost_equal(actual, expected, 2)

    def test_correctness_fast_treeshap_v1(self, data, fast_treeshap_v1, treeshap, tree):
        x, y, x_test = data
        actual = fast_treeshap_v1.shap_values(x_test)
        expected = treeshap.shap_values(x_test)
        np.testing.assert_array_almost_equal(actual, expected, 2)

    def test_correctness_fast_treeshap_v2(self, data, fast_treeshap_v2, treeshap, tree):
        x, y, x_test = data
        actual = fast_treeshap_v2.shap_values(x_test)
        expected = treeshap.shap_values(x_test)
        np.testing.assert_array_almost_equal(actual, expected, 2)

class TestConductor(TestAdult):
    @pytest.fixture
    def data(self):
        return load_conductor()
