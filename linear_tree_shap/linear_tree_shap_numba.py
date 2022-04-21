import numpy as np
from utils import copy_tree, get_N_prime
import numpy as np
import numba 

#@numba.njit
def polymul(X, a, Out, Out_len):
    Out[0] = X[0]
    for i in range(1, Out_len-1):
        Out[i] = X[i] + a*X[i-1]
    Out[Out_len-1] = a*X[Out_len-2]

#@numba.njit
def polyquo(X, a, Out, Out_len):
    print(a)
    Out[0] = X[0]
    for i in range(1, Out_len):
        Out[i] =  X[i]- Out[i-1]*a

#@numba.njit
def psi(E, q, n, e_size, d):
    res = 0
    quo = 0
    for i in range(d+1):
        if i < e_size:
            quo = E[i] - quo*q
        else:
            quo = - quo*q
        res += quo*n[d-i]
    return res/(d+1)


#@numba.jit
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
              result, 
              C, E, N, node=0, 
              edge_feature=-1, 
              depth=0, c_size=1):

    left, right, parent, child_edge_feature = (
                            children_left[node], 
                            children_right[node], 
                            parents[node], 
                            features[node]
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
            q_eff = 1./weights[node] - 1.
        else:
            q_eff = -1.

        polymul(C[depth-1], q_eff, C[depth], c_size)

        if parent >= 0:
            if activation[parent]:
                s_eff = 1./weights[parent] - 1.
            else:
                s_eff = -1.
            c_size -= 1
            polyquo(C[depth], s_eff, C[depth], c_size)

    if left < 0:
        E[depth, :edge_heights[node]+1] = C[depth, :c_size]*leaf_predictions[node]
    else:
        if edge_heights[left] > edge_heights[right]:
            first, second = left, right
        else:
            first, second = right, left
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
                  result, 
                  C, E, N, first, 
                  child_edge_feature, 
                  depth+1, c_size+1)
        E[depth, :edge_heights[first]+1] = E[depth+1, :edge_heights[first]+1]
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
                  result, 
                  C, E, N, second, 
                  child_edge_feature, 
                  depth+1, c_size+1)
        E[depth, :edge_heights[second]+1] += E[depth+1, :edge_heights[second]+1]
    current_height = edge_heights[node]
    if edge_feature >= 0:
        value = q_eff*psi(E[depth], q_eff, N[current_height-1], current_height+1, current_height-1)
        result[edge_feature] += value
        if parent >= 0:
            parent_height = edge_heights[parent]
            result[edge_feature] -= s_eff*psi(E[depth], s_eff, N[parent_height-1], current_height+1, parent_height-1)

def fast_inference(tree, C, E, N, result, activation, x):
    for i in range(x.shape[0]):
        _inference(tree.weights,
                   tree.leaf_predictions,
                   tree.parents, 
                   tree.edge_heights,
                   tree.features,
                   tree.children_left, 
                   tree.children_right, 
                   tree.thresholds, 
                   tree.max_depth, 
                   x[i],
                   activation, 
                   result[i], 
                   C, E, N)

def inference(tree, x):
    C = np.zeros((tree.max_depth, tree.max_depth))
    E = np.zeros((tree.max_depth, tree.max_depth))
    N = get_N_prime(tree.max_depth)
    result = np.zeros_like(x)
    activation = np.zeros_like(tree.children_left, dtype=bool)
    C[0, 0] = 1.
    fast_inference(tree, C, E, N, result, activation, x)
    # for i in range(x.shape[0]):
    #     _inference(tree.weights,
    #                tree.leaf_predictions,
    #                tree.parents, 
    #                tree.edge_heights,
    #                tree.features,
    #                tree.children_left, 
    #                tree.children_right, 
    #                tree.thresholds, 
    #                tree.max_depth, 
    #                x[i],
    #                activation, 
    #                result[i], 
    #                C, E, N)
    return result

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, export_text
    from shap import TreeExplainer as Truth
    import numpy as np
    np.random.seed(10)
    x, y = make_regression(1000, n_features=10)
    clf = DecisionTreeRegressor(max_depth=6).fit(x, y)
    sim = Truth(clf)
    mine_tree = copy_tree(clf.tree_)
    result = inference(mine_tree, x[:2])
    b = sim.shap_values(x[:2])
    np.testing.assert_array_almost_equal(result, b, 3)
