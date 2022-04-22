import numpy as np
from utils import copy_tree
import scipy.special as sp
import numpy as np
import numba 
import time

np.seterr(divide='raise')

@numba.jit
def psi(E, D, q, Ns, d):
    n = Ns[d, :d]
    return ((E/(D+q))[:d]).dot(n)/d

@numba.jit
def _inference(weights,
              leaf_predictions,
              parents, 
              edge_heights,
              features,
              children_left, 
              children_right, 
              thresholds, 
              max_depth, 
              x, 
              activation, 
              result, D_powers, 
              D, Ns, C, E, node=0, 
              edge_feature=-1, depth=0):

    left, right, parent, child_edge_feature = (
                            children_left[node], 
                            children_right[node], 
                            parents[node], 
                            features[node]
                            )
    left_height, right_height, parent_height, current_height = ( 
                            edge_heights[left], 
                            edge_heights[right], 
                            edge_heights[parent], 
                            edge_heights[node]
                            )
    if left >= 0:
        if x[child_edge_feature] <= thresholds[node]:
            activation[left], activation[right] = True, False
        else:
            activation[left], activation[right] = False, True

    if edge_feature >= 0:
        if parent >= 0:
            activation[node] &= activation[parent]

        if activation[node]:
            q_eff = 1./weights[node]
        else:
            q_eff = 0. 
        C[depth] = C[depth-1]*(D+q_eff)

        if parent >= 0:
            if activation[parent]:
                s_eff = 1./weights[parent]
            else:
                s_eff = 0.
            C[depth] = C[depth]/(D+s_eff)
    if left < 0:
        E[depth] = C[depth]*leaf_predictions[node]
    else:
        _inference(weights,
                  leaf_predictions,
                  parents, 
                  edge_heights,
                  features,
                  children_left, 
                  children_right, 
                  thresholds, 
                  max_depth, 
                  x,
                  activation, 
                  result, D_powers,
                  D, Ns, C, E, left, 
                  child_edge_feature, 
                  depth+1
                  )
        E[depth] = E[depth+1]*D_powers[current_height-left_height]
        _inference(weights,
                  leaf_predictions,
                  parents, 
                  edge_heights,
                  features,
                  children_left, 
                  children_right, 
                  thresholds, 
                  max_depth, 
                  x,
                  activation, 
                  result, D_powers,
                  D, Ns, C, E, right, 
                  child_edge_feature,
                  depth+1
                  )
        E[depth] += E[depth+1]*D_powers[current_height-right_height]


    if edge_feature >= 0:
        value = (q_eff-1)*psi(E[depth], D, q_eff, Ns, current_height)
        result[edge_feature] += value
        if parent >= 0:
            value = (s_eff-1)*psi(E[depth]*D_powers[parent_height-current_height], D, s_eff, Ns, parent_height)
            result[edge_feature] -= value

@numba.jit
def fast_inference(tree, D, D_powers, Ns, result, activation, x, max_depth, C, E):
    for i in range(x.shape[0]):
        _inference(tree.weights,
                   tree.leaf_predictions,
                   tree.parents, 
                   tree.edge_heights,
                   tree.features,
                   tree.children_left, 
                   tree.children_right, 
                   tree.thresholds, 
                   max_depth, 
                   x[i],
                   activation, 
                   result[i], D_powers,
                   D, Ns, C, E)

def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])

def get_N(D):
    depth = D.shape[0]
    Ns = np.zeros((depth+1, depth))
    for i in range(1, depth+1):
        Ns[i,:i] = np.linalg.inv(np.vander(D[:i]).T).dot(1./get_norm_weight(i-1)) 
    return Ns

def cache(D):
    return np.vander(D+1).T[::-1]

def inference(tree, x):
    D = np.polynomial.chebyshev.chebpts2(tree.max_depth)
    D_powers = cache(D)
    Ns = get_N(D)
    activation = np.zeros_like(tree.children_left, dtype=bool)
    C = np.zeros((tree.max_depth+1, tree.max_depth))
    E = np.zeros((tree.max_depth+1, tree.max_depth))
    C[0, :] = 1
    result = np.zeros_like(x)
    fast_inference(tree, D, D_powers, Ns, result, activation, x, tree.max_depth, C, E)
    result = np.zeros_like(x)
    start = time.time()
    fast_inference(tree, D, D_powers, Ns, result, activation, x, tree.max_depth, C, E)
    print('mine', time.time()-start)
    return result

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, export_text
    from shap import TreeExplainer as Truth
    import numpy as np
    np.random.seed(10)
    x, y = make_regression(10000, n_features=100)
    max_depth = 25
    clf = DecisionTreeRegressor(max_depth=max_depth).fit(x, y)
    sim = Truth(clf)
    mine_tree = copy_tree(clf.tree_)
    result = inference(mine_tree, x[:10])
    start = time.time()
    b = sim.shap_values(x[:10])
    print('b', time.time()-start)
    print('mine', y[:10]-result.sum(axis=1))
    print('b', y[:10]-b.sum(axis=1))
    np.testing.assert_array_almost_equal(result, b, 1)
