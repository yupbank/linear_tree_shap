from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import scipy.special as sp
from collections import namedtuple
from functools import reduce

Interval = namedtuple('Interval', 'lower_bound upper_bound weight')
DEFAULT_INTERVAL = Interval(-np.inf, np.inf, 1.0)

def N(m):
    return np.array([sp.binom(m, i) for i in range(m + 1)])

def merge_interval(new_interval, original_interval=None):
    original_interval = original_interval or DEFAULT_INTERVAL
    return Interval(
        max(original_interval.lower_bound, new_interval.lower_bound),
        min(original_interval.upper_bound, new_interval.upper_bound),
        original_interval.weight * new_interval.weight
    )


def collect_branches(tree, branch_info=None, node=0, branches=None):
    branch_info = branch_info if branch_info is not None else dict()
    branches = branches if branches is not None else []
    n_samples = tree.n_node_samples[node]
    feature_index = tree.feature[node]
    threshold = tree.threshold[node]
    children_left = tree.children_left[node]
    children_right = tree.children_right[node]
    if children_left == -1 or children_right == -1:
        # reaching a leaf
        value = tree.value[node].ravel()[0]
        branches.append((branch_info, value, node))
    else:
        if children_left != -1:
            left_branch_info = branch_info.copy()
            left_weight = tree.n_node_samples[children_left] / n_samples
            left_interval = Interval(-np.inf, threshold, left_weight)
            left_branch_info[feature_index] = merge_interval(
                left_interval, left_branch_info.get(feature_index))
            collect_branches(tree, left_branch_info, children_left, branches)

        if children_right != -1:
            right_branch_info = branch_info.copy()
            right_weight = tree.n_node_samples[children_right] / n_samples
            right_interval = Interval(threshold, np.inf, right_weight)
            right_branch_info[feature_index] = merge_interval(
                right_interval, right_branch_info.get(feature_index))
            collect_branches(tree, right_branch_info, children_right, branches)
    return branches

def psi(E, q, f):
    poly, rem = np.polydiv(E, q)
    degree = poly.shape[0]
    return poly.dot(1./N(degree-1))/degree

def shapely_value(branches, x, values):
    for branch_info, leaf_value, node in branches:
        features, upper_bounds, lower_bounds, weights = [], [], [], []
        for feature, interval in branch_info.items():
            features.append(feature)
            upper_bounds.append(interval.upper_bound)
            lower_bounds.append(interval.lower_bound)
            weights.append(interval.weight)
        features = np.array(features)
        upper_bounds = np.array(upper_bounds)
        lower_bounds = np.array(lower_bounds)
        weights = np.array(weights)
        Be = np.prod(weights)*leaf_value
        in_branch = np.logical_and(lower_bounds <= x[features], x[features] < upper_bounds)
        Q = np.where(in_branch, 1/weights, 0)
        Q = np.vstack([np.ones_like(Q), Q]).T
        E = reduce(np.polymul, Q)*Be
        for f, q in zip(features, Q):
            values[f] += (q[1]-1)*psi(E, q, f)

def truth_shap(clf, x):
    branches = collect_branches(clf.tree_)
    values = np.zeros_like(x[0])
    shapely_value(branches, x[0], values)
    return values

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    from shap import TreeExplainer as Truth
    np.random.seed(10)
    x, y = make_regression(10000, n_features=100)
    clf = DecisionTreeRegressor(max_depth=16).fit(x, y)
    sim = Truth(clf)
    branches = collect_branches(clf.tree_)
    a = np.zeros_like(x[12])
    b = sim.shap_values(x[12][None,:])[0]
    shapely_value(branches, x[12], a)
    np.testing.assert_array_almost_equal(a, b, 5)
    print(a[44])
