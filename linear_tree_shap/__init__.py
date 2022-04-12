import numpy as np
from functools import reduce
from utils import copy_tree, get_N_prime


def get_activation(tree, x):
    edge_active = np.zeros((x.shape[0], tree.children_left.shape[0]), dtype=bool)
    node_mask = tree.features >= 0
    edge_active[:, tree.children_left[node_mask]] = x[:, tree.features[node_mask]] <= tree.thresholds[node_mask]
    edge_active[:, tree.children_right[node_mask]] = x[:, tree.features[node_mask]] > tree.thresholds[node_mask]
    return edge_active

   
def oplus(El, Er):
    dl = El.shape[0]
    dr = Er.shape[0]
    if dl > dr:
        return oplus(Er, El)
    return np.hstack([El, np.zeros(dr-dl)])+Er

def polymul(C, q, out):
    tmp = 0
    for i in range(C.shape[0]):
        out[i] = C[i] + q*tmp
        tmp = C[i]
    out[C.shape[0]] = q*tmp

def polyquo(C, q, out):
    quo = 0
    for i in range(C.shape[0]-1):
        quo = C[i] - quo*q
        out[i] = quo

def psi(E, q, e_size, d, N):
    n = N[d, :d+1]
    res = quo = 0
    for i in range(d+1):
        if i < e_size:
            quo = E[i] - quo*q
        else:
            quo = 0 - quo*q
        res += quo*n[d-i]
    return res/(d+1)

def inference(tree, A, V, N):
    C = np.zeros((tree.max_depth, tree.max_depth))
    E = np.zeros((tree.max_depth, tree.max_depth))
    C[0, 0] = 1.
    def _inference(n=0, feature=-1, depth=0, c_size=0):
        q = -1
        s = -1
        m = tree.parents[n]
        left, right = tree.children_left[n], tree.children_right[n]
        c_size += 1
        if feature >= 0:
            if m >= 0:
                A[n] &= A[m]
                if A[m]:
                    s = 1/tree.weights[m] - 1
            if A[n]:
                q = 1/tree.weights[n] - 1
            polymul(C[depth-1, :c_size-1], q, C[depth, :c_size])
            if m >= 0:
                c_size -= 1
                polyquo(C[depth, :c_size+1], s, C[depth, :c_size])
        if left >= 0:
            if tree.edge_heights[left] > tree.edge_heights[right]:
                first, second = left, right
            else:
                first, second = right, left
            E_l = _inference(first, tree.features[n], depth+1, c_size)
            E[depth, :tree.edge_heights[first]+1] = E[depth+1, :tree.edge_heights[first]+1]
            E_r = _inference(second, tree.features[n], depth+1, c_size)
            E[depth, :tree.edge_heights[n]+1] = oplus(E[depth, :tree.edge_heights[first]+1], 
                                                      E[depth+1, :tree.edge_heights[second]+1])
        else:
            E[depth, :tree.edge_heights[n]+1] = tree.leaf_predictions[n]*C[depth, :c_size]
        e = E[depth, :tree.edge_heights[n]+1]
        if feature >= 0:
            add_value = q*psi(e, q, tree.edge_heights[n]+1, tree.edge_heights[n]-1, N)
            #print(n, add_value, e, q)
            if n==126:
                print(A[n], m, A[m])
            V[feature] += add_value
            if m >= 0:
                remove_value = s*psi(e, s, tree.edge_heights[n]+1, tree.edge_heights[m]-1, N)
                V[feature] -= remove_value
        return E
    _inference()
    return V
    
class TreeExplainer:
    def __init__(self, clf):
        self.clf = clf
        self.tree = copy_tree(clf.tree_)
        self.N = get_N_prime(self.tree.max_depth)


    def py_shap_values(self, x):
        V = np.zeros_like(x)
        A = get_activation(self.tree, x)
        return inference(self.tree, A[0], V[0], self.N)

    def shap_values(self, X):
        from linear_tree_shap import _cext
        V = np.zeros_like(X)
        _cext.linear_tree_shap(
                               self.tree.weights, 
                               self.tree.leaf_predictions, 
                               self.tree.thresholds, 
                               self.tree.parents.astype(np.int32), 
                               self.tree.edge_heights.astype(np.int32), 
                               self.tree.features.astype(np.int32), 
                               self.tree.children_left.astype(np.int32), 
                               self.tree.children_right.astype(np.int32), 
                               self.tree.max_depth,
                               self.tree.num_nodes,
                               self.N, X, V)
        return V


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, export_text
    from shap import TreeExplainer as Truth
    import numpy as np
    np.random.seed(10)
    x, y = make_regression(1000, n_features=10)
    clf = DecisionTreeRegressor(max_depth=6).fit(x, y)
    sim = Truth(clf)
    mine = TreeExplainer(clf)
    a = mine.py_shap_values(x[4][None, :])
    b = sim.shap_values(x[4][None,:])[0]
    np.testing.assert_array_almost_equal(a, b, 3)
