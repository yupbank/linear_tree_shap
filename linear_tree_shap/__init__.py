import numpy as np
import scipy.special as sp
from collections import namedtuple
from functools import reduce
Tree = namedtuple('Tree', 
                'weights,leaf_predictions,parents,edge_heights,features,children_left,children_right,thresholds,max_depth,num_nodes')

def copy_tree(tree):
    weights = np.ones_like(tree.threshold)
    leaf_predictions = np.zeros_like(tree.threshold)
    parents = np.full_like(tree.children_left, -1)
    edge_heights = np.zeros_like(tree.children_left)
    
    def _recursive_copy(node=0, feature=None, 
                        parent_samples=None, prod_weight=1.0, 
                        seen_features=dict()):
        n_sample, child_left, child_right = (tree.n_node_samples[node],
                            tree.children_left[node], tree.children_right[node])
        if feature is not None:
            weight = n_sample/parent_samples
            prod_weight *= weight
            if feature in seen_features:
                parents[node] = seen_features[feature]
                weight *= weights[seen_features[feature]]
            weights[node] = weight
            seen_features[feature] = node
        if child_left >= 0: # not leaf
            left_max_features = _recursive_copy(child_left, tree.feature[node], n_sample, prod_weight, seen_features.copy())
            right_max_features = _recursive_copy(child_right, tree.feature[node], n_sample, prod_weight, seen_features.copy())
            edge_heights[node] = max(left_max_features, right_max_features)
            return edge_heights[node]
        else:               # is leaf
            leaf_predictions[node] =  prod_weight*tree.value[node].ravel()[0]
            edge_heights[node] = len(seen_features)
            return edge_heights[node]
    _recursive_copy()
    return Tree(weights, leaf_predictions, parents, edge_heights, tree.feature, tree.children_left, tree.children_right, tree.threshold, tree.max_depth+2, tree.children_left.shape[0])


def get_activation(tree, x):
    edge_active = np.zeros((x.shape[0], tree.children_left.shape[0]), dtype=bool)
    node_mask = tree.features >= 0
    edge_active[:, tree.children_left[node_mask]] = x[:, tree.features[node_mask]] <= tree.thresholds[node_mask]
    edge_active[:, tree.children_right[node_mask]] = x[:, tree.features[node_mask]] > tree.thresholds[node_mask]
    return edge_active

def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])

def get_N(max_size=10):
    N = np.zeros((max_size, max_size))
    for i in range(max_size):
        N[i,:i+1] = get_norm_weight(i)
    return N

def get_N_prime(max_size=10):
    N = np.zeros((max_size, max_size))
    for i in range(max_size):
        N[i,:i+1] = get_norm_weight(i)
    N_prime = np.zeros((max_size, max_size))
    for i in range(max_size):
        N_prime[i,:i+1] = N[:i+1, :i+1].dot(1/N[i, :i+1])
    return N_prime
   
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
        c = C[depth-1, :c_size]
        c_size += 1
        if feature >= 0:
            if m >= 0:
                A[n] &= A[m]
                if A[m]:
                    s = 1/tree.weights[m] - 1
            if A[n]:
                q = 1/tree.weights[n] - 1
            polymul(c, q, C[depth, :c_size])
            if m >= 0:
                c_size -= 1
                polyquo(C[depth, :c_size+1], s, C[depth, :c_size])
        if left >= 0:
            E_l = _inference(left, tree.features[n], depth+1, c_size)
            E[depth, :tree.edge_heights[left]+1] = E[depth+1, :tree.edge_heights[left]+1]
            E_r = _inference(right, tree.features[n], depth+1, c_size)
            E[depth, :tree.edge_heights[n]+1] = oplus(E[depth, :tree.edge_heights[left]+1], 
                                                      E[depth+1, :tree.edge_heights[right]+1])
        else:
            E[depth, :tree.edge_heights[n]+1] = tree.leaf_predictions[n]*C[depth, :c_size]
        e = E[depth, :tree.edge_heights[n]+1]
        if feature >= 0:
            V[feature] += q*psi(e, q, tree.edge_heights[n]+1, tree.edge_heights[n]-1, N)
            if m >= 0:
                V[feature] -= s*psi(e, s, tree.edge_heights[n]+1, tree.edge_heights[m]-1, N)
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
    clf = DecisionTreeRegressor(max_depth=13).fit(x, y)
    sim = Truth(clf)
    mine = TreeExplainer(clf)
    a = mine.py_shap_values(x[0:1])
    b = sim.shap_values(x[0:1])[0]
    np.testing.assert_array_almost_equal(a, b, 5)
