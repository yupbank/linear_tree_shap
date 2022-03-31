#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "linear_tree_shap.h"

static PyObject *_cext_linear_tree_shap(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"linear_tree_shap", _cext_linear_tree_shap, METH_VARARGS, "C implementation of Linear Tree SHAP."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cext",
    "This module provides an interface for a linear Tree SHAP implementation.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__cext(void)
#else
PyMODINIT_FUNC init_cext(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create(&moduledef);
        if (!module) return NULL;
    #else
        PyObject *module = Py_InitModule("_cext", module_methods);
        if (!module) return;
    #endif

    /* Load `numpy` functionality. */
    import_array();

    #if PY_MAJOR_VERSION >= 3
        return module;
    #endif
}


static PyObject *_cext_linear_tree_shap(PyObject *self, PyObject *args)
{
    PyObject *weights_obj; 
    PyObject *leaf_predictions_obj;
    PyObject *thresholds_obj;
    PyObject *parents_obj; 
    PyObject *edge_heights_obj; 
    PyObject *features_obj; 
    PyObject *children_left_obj; 
    PyObject *children_right_obj;
    int max_depth;
    int num_nodes;
    PyObject *norm_obj; 
    PyObject *X_obj;
    PyObject *out_contribs_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOOOiiOOO", 
	&weights_obj, 
	&leaf_predictions_obj,
	&thresholds_obj,
	&parents_obj, 
	&edge_heights_obj, 
    &features_obj, 
	&children_left_obj, 
	&children_right_obj,
    &max_depth,
    &num_nodes,
	&norm_obj, 
	&X_obj,
    &out_contribs_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *weights_array = (PyArrayObject*)PyArray_FROM_OTF(weights_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *leaf_predictions_array = (PyArrayObject*)PyArray_FROM_OTF(leaf_predictions_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *parents_array = (PyArrayObject*)PyArray_FROM_OTF(parents_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *edge_heights_array = (PyArrayObject*)PyArray_FROM_OTF(edge_heights_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);

    PyArrayObject *norm_array = (PyArrayObject*)PyArray_FROM_OTF(norm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_contribs_array = (PyArrayObject*)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. Note that R and y are optional. */
    if (children_left_array == NULL || children_right_array == NULL ||
        features_array == NULL || leaf_predictions_array == NULL || 
	thresholds_array == NULL ||
        edge_heights_array == NULL || parents_array == NULL || 
	weights_array == NULL || norm_array == NULL ||
	X_array == NULL || out_contribs_array == NULL
	) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(features_array);
        Py_XDECREF(leaf_predictions_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(edge_heights_array);
        Py_XDECREF(parents_array);
        Py_XDECREF(weights_array);
        Py_XDECREF(norm_array);
        Py_XDECREF(X_array);
        //PyArray_ResolveWritebackIfCopy(out_contribs_array);
        Py_XDECREF(out_contribs_array);
        return NULL;
    }

    // Get pointers to the data as C-types
    tfloat *weights = (tfloat*)PyArray_DATA(weights_array);
    tfloat *leaf_predictions  = (tfloat*)PyArray_DATA(leaf_predictions_array);
    tfloat *thresholds  = (tfloat*)PyArray_DATA(thresholds_array);
    int *parents = (int*)PyArray_DATA(parents_array);
    int *edge_heights = (int*)PyArray_DATA(edge_heights_array);
    int *features = (int*)PyArray_DATA(features_array);
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    tfloat *norm  = (tfloat*)PyArray_DATA(norm_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    tfloat *out_contribs = (tfloat*)PyArray_DATA(out_contribs_array);

    // these are just a wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    Tree tree = Tree(
    	weights, leaf_predictions,
	thresholds,
	parents, edge_heights,
	features, 
        children_left, children_right,
        max_depth, num_nodes
    );
    //const unsigned num_x = pyarray_dim(a_array, 0);
    //const unsigned m = pyarray_dim(a_array, 1);
    //Activation data = Activation(A, num_X, M);

    const unsigned num_n = PyArray_DIM(norm_array, 0);
    const unsigned n = PyArray_DIM(norm_array, 1);
    BioCoeff N = BioCoeff(norm, num_n, n);
    const unsigned row_x = PyArray_DIM(X_array, 0);
    const unsigned col_x = PyArray_DIM(X_array, 1);
    Dataset data = Dataset(X, row_x, col_x);

    const unsigned row_out = PyArray_DIM(out_contribs_array, 0);
    const unsigned col_out = PyArray_DIM(out_contribs_array, 1);
    Dataset out = Dataset(out_contribs, row_out, col_out);
    linear_tree_shap(tree, data, out, N);

    // retrieve return value before python cleanup of objects
    tfloat ret_value = (double)leaf_predictions[0];
    // clean up the created python objects 
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(features_array);
    Py_XDECREF(leaf_predictions_array);

    Py_XDECREF(edge_heights_array);
    Py_XDECREF(parents_array);
    Py_XDECREF(weights_array);

    Py_XDECREF(X_array);
    Py_XDECREF(out_contribs_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", ret_value);
    return ret;
}
