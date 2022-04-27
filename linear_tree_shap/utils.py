import numpy as np
from collections import namedtuple
import scipy.special as sp

Tree = namedtuple('Tree', 'weights,leaf_predictions,parents,edge_heights,features,children_left,children_right,thresholds,max_depth,num_nodes')

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
            edge_heights[node] = len(seen_features)
            return edge_heights[node]
    _recursive_copy()
    return Tree(weights, tree.n_node_samples/tree.n_node_samples[0]*tree.value.ravel(), parents, edge_heights, tree.feature, tree.children_left, tree.children_right, tree.threshold, tree.max_depth, tree.children_left.shape[0])


def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])

def get_N(max_size=10):
    N = np.zeros((max_size, max_size))
    for i in range(max_size):
        N[i,:i+1] = get_norm_weight(i)
    return N

def get_N_prime(max_size=10):
    N = np.zeros((max_size+2, max_size+2))
    for i in range(max_size+2):
        N[i,:i+1] = get_norm_weight(i)
    N_prime = np.zeros((max_size+2, max_size+2))
    for i in range(max_size+2):
        N_prime[i,:i+1] = N[:i+1, :i+1].dot(1/N[i, :i+1])
    return N_prime

def get_N_v2(D):
    depth = D.shape[0]
    Ns = np.zeros((depth+1, depth))
    for i in range(1, depth+1):
        Ns[i,:i] = np.linalg.inv(np.vander(D[:i]).T).dot(1./get_norm_weight(i-1)) 
    return Ns
