#include <stdio.h> 
#include <cmath>

#if defined(_WIN32) || defined(WIN32)
    #include <malloc.h>
#elif defined(__MVS__)
    #include <stdlib.h>
#else
    #include <alloca.h>
#endif
using namespace std;

typedef double tfloat;

struct Tree {
    tfloat *weights;
    tfloat *leaf_predictions;
    tfloat *thresholds;
    int *parents;
    int *edge_heights;
    int *features;
    int *children_left;
    int *children_right;
    int  max_depth;
    int  num_nodes;

    Tree(tfloat *weights, tfloat *leaf_predictions, tfloat *thresholds,
    		 int *parents, int *edge_heights,
		 int *features, int *children_left, int *children_right, int max_depth, int num_nodes):
        weights(weights), leaf_predictions(leaf_predictions), thresholds(thresholds),
        parents(parents), edge_heights(edge_heights),
	    features(features), children_left(children_left),
	children_right(children_right), max_depth(max_depth), num_nodes(num_nodes){};

    bool is_internal(int pos)const {
        return children_left[pos] >= 0;
    }
};

struct BioCoeff {
    tfloat *C;
    unsigned row;
    unsigned col;
    BioCoeff() {};
    BioCoeff(tfloat *C, unsigned row, unsigned col):
    C(C), row(row), col(col){};
    tfloat * get_n(const int i) const{
        return C+i*col;
    }

    tfloat psi(tfloat* E, tfloat q, int e_size, int d) {
	 tfloat res=0., quo = 0.;
	 tfloat *n = get_n(d);
	 for(int i=0; i<d+1; i++){
		if (i<e_size){
		    quo = E[i] - quo*q;
		} else{
		    quo = - quo*q;
		}
		res += quo*n[d-i];
	 }
	 return res/(d+1);
    }
};

struct Dataset {
    tfloat *X;
    unsigned row;
    unsigned col;
    Dataset() {};
    Dataset(tfloat *X, unsigned row, unsigned col):
    X(X), row(row), col(col){};
    tfloat * get_n(const unsigned i){
        return X+i*col;
    }
};

void polymul(tfloat * C, tfloat q, int c_size, tfloat * out){
    tfloat tmp = 0.;
    for(int i=0; i<c_size; i++){
	    out[i] = C[i] + q*tmp;
	    tmp = C[i];
    }
    out[c_size] = q*tmp;
};

void polyquo(tfloat * C, tfloat q, int c_size, tfloat * out){
    tfloat tmp = 0.;
    for(int i=0; i<c_size-1; i++){
	    tmp = C[i] - q*tmp;
	    out[i] = tmp;
    }
};

void multiply(tfloat * input, tfloat *output, tfloat scalar, int size){
    for(int i =0; i<size; i++){
	    output[i] = input[i]*scalar;
    }
};

void sum(tfloat * input, tfloat* output, int size){
    for(int i =0; i<size; i++){
	    output[i] += input[i];
    }
};

void copy(tfloat *from, tfloat *to, int size){
    for (int i=0; i<size; i++){
	    to[i] = from[i];
    }
};


void inference(const Tree& tree, 
	      bool *A, 
	      tfloat* V, 
	      BioCoeff& N, 
	      tfloat* C, 
	      tfloat* E, 
	      tfloat* x,
	      int n=0,
	      int feature=-1,
	      int depth=0,              
	      int prev_c_size=0){
    tfloat q = -1.;
    tfloat s = -1.;
    int m = tree.parents[n];
    int left = tree.children_left[n];
    int right = tree.children_right[n];
    tfloat *current_c = C+depth*tree.max_depth;
    tfloat *current_e = E+depth*tree.max_depth;
    tfloat *child_e = E+(depth+1)*tree.max_depth;
    int current_c_size = prev_c_size + 1;
    if(x[tree.features[n]] <= tree.thresholds[n]){
	 A[left] = true;
	 A[right] = false;
    } else{
	 A[left] = false;
	 A[right] = true;
    }
    if (feature >= 0){
         if (m >= 0) {
             A[n] = A[n] & A[m];
             if (A[m]){
                s = 1/tree.weights[m]-1;
             }
         } 
         if (A[n]) {
             q = 1/tree.weights[n]-1;
         }
	     tfloat *prev_c = C+(depth-1)*tree.max_depth;
         polymul(prev_c, q, prev_c_size, current_c);

	     if (m >= 0){
           polyquo(current_c, s, current_c_size, current_c);
	       current_c_size -= 1;
	     }
    }
    if (tree.is_internal(n)){
            int first, second;
            if(tree.edge_heights[left] > tree.edge_heights[right]){
                first = left;
                second = right;
                }else{
                  first = right;
                  second = left;
                };
            inference(tree, A, V, N, C, E, x, first, tree.features[n], depth+1, current_c_size);
	        copy(child_e, current_e, tree.edge_heights[first]+1);
            inference(tree, A, V, N, C, E, x, second, tree.features[n], depth+1, current_c_size);
            sum(child_e, current_e, tree.edge_heights[second]+1);
    } else {
            multiply(current_c, current_e, tree.leaf_predictions[n], current_c_size);
    }
    if (feature >= 0){
        V[feature] += q*N.psi(current_e, q, tree.edge_heights[n]+1, tree.edge_heights[n]-1);
        if (m >= 0){
            V[feature] -= s*N.psi(current_e, s, tree.edge_heights[n]+1, tree.edge_heights[m]-1);
        }
    }
};


inline void linear_tree_shap(const Tree& tree, 
			     Dataset& data, 
			     Dataset& out,
               		     BioCoeff& N){
    tfloat * C = new tfloat[tree.max_depth*tree.max_depth];
    tfloat * E = new tfloat[tree.max_depth*tree.max_depth];
    bool *A = new bool[tree.num_nodes];
    for (unsigned i=0; i<data.row; i++)
    {
      C[0] = 1.0;
      tfloat *x = data.get_n(i);
      tfloat *o = out.get_n(i);
      inference(tree, A, o, N, C, E, x);
    }
    delete[] C;
    delete[] E;
    delete[] A;
};
