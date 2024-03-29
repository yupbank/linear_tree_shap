import numpy as np
from functools import reduce
from linear_tree_shap.utils import copy_tree, get_N_prime, get_N_v2


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

def psi_v2(E, D_power, D, q, Ns, d):
    n = Ns[d, :d]
    return ((E*D_power/(D+q))[:d]).dot(n)/d

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
   

def inference_v2(tree, x, activation, result, Base, Offset, Ns, C, E, node=0, edge_feature=-1, depth=0):
    left, right, parent, child_edge_feature = (
                            tree.children_left[node], 
                            tree.children_right[node], 
                            tree.parents[node], 
                            tree.features[node]
                            )
    left_height, right_height, parent_height, current_height = ( 
                            tree.edge_heights[left], 
                            tree.edge_heights[right], 
                            tree.edge_heights[parent], 
                            tree.edge_heights[node]
                            )
    if left >= 0:
        if x[child_edge_feature] <= tree.thresholds[node]:
            activation[left], activation[right] = True, False
        else:
            activation[left], activation[right] = False, True
    if edge_feature >= 0:
        if parent >= 0:
            activation[node] &= activation[parent]

        if activation[node]:
            q_eff = 1./tree.weights[node]
        else:
            q_eff = 0. 
        C[depth] = C[depth-1]*(Base+q_eff)

        if parent >= 0:
            if activation[parent]:
                s_eff = 1./tree.weights[parent]
            else:
                s_eff = 0.
            C[depth] = C[depth]/(Base+s_eff)
    if left < 0:
        E[depth] = C[depth]*tree.leaf_predictions[node]
    else:
        inference_v2(tree, x, activation, result, Base, Offset, Ns, C, E, left, child_edge_feature, depth+1)
        E[depth] = E[depth+1]*Offset[current_height-left_height]
        inference_v2(tree, x, activation, result, Base, Offset, Ns, C, E, right, child_edge_feature, depth+1)
        E[depth] += E[depth+1]*Offset[current_height-right_height]
    if edge_feature >= 0:
        value = (q_eff-1)*psi_v2(E[depth], Offset[0], Base, q_eff, Ns, current_height)
        result[edge_feature] += value
        if parent >= 0:
            value = (s_eff-1)*psi_v2(E[depth], Offset[parent_height-current_height], Base, s_eff, Ns, parent_height)
            result[edge_feature] -= value

class TreeExplainer:
    def __init__(self, clf, base_func=np.polynomial.chebyshev.chebpts2):
        self.clf = clf
        self.tree = copy_tree(clf.tree_)
        self.N = get_N_prime(self.tree.max_depth)
        self.Base = base_func(self.tree.max_depth)
        self.Offset = np.vander(self.Base+1).T[::-1]
        self.N_v2 = get_N_v2(self.Base)

    def py_shap_values(self, x):
        V = np.zeros_like(x)
        A = get_activation(self.tree, x)
        return inference(self.tree, A[0], V[0], self.N)

    def py_shap_values_v2(self, x):
        activation = np.zeros_like(self.tree.children_left, dtype=bool)
        C = np.zeros((self.tree.max_depth+1, self.tree.max_depth))
        E = np.zeros((self.tree.max_depth+1, self.tree.max_depth))
        C[0, :] = 1
        result = np.zeros_like(x)
        inference_v2(self.tree, x.astype(np.float32), activation, result, self.Base, self.Offset, self.N_v2, C, E)
        return result

    def shap_values(self, X):
        from linear_tree_shap import _cext
        V = np.zeros_like(X, dtype=np.float64)
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
                               self.N, X.astype(np.float32), V)
        return V
    
    def shap_values_v2(self, X):
        from linear_tree_shap import _cext
        V = np.zeros_like(X, dtype=np.float64)
        _cext.linear_tree_shap_v2(
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
                               self.Base, self.Offset, self.N_v2, X.astype(np.float32), V)
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
    a = mine.py_shap_values_v2(x[4])
    b = sim.shap_values(x[4][None,:])[0]
    np.testing.assert_array_almost_equal(a, b, 3)
