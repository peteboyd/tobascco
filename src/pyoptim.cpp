#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <iostream>
#include <map>
#include <math.h>
#include <nlopt.h>
#include <numpy/arrayobject.h>

void create_full_rep(int, int, double**, int, int, const double*, double**);
static PyObject * nloptimize(PyObject *self, PyObject *args);
double objectivefunc(unsigned, const double*, double*, void*);
double sumsquarediff(int, int*, int*, double**, double**);
void matrix_multiply(int, int, double **, int, int, double **, double**);
void create_metric_tensor(int, const double*, double**);
void transpose_2darray(int, int, double**, double**);
void setup_matrix(int, int, double**);
double * get1darrayd(PyArrayObject*);
int * get1darrayi(PyArrayObject*);
double ** get2darrayd(PyArrayObject*);
double ** construct2darray(int rows, int cols);
void free_2d_array(double**, int);
int factorial(int, int);

static PyMethodDef functions[] = {
    {"nloptimize", nloptimize, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL} 
};


struct data_info{
    int rep_size, cycle_size, nz_size, x_size;
    int B_shape, ndim, diag_ind;
    int angle_inds, start;
    double *lb, *ub;
    int *_zi, *_zj;
    double ** _cycle_cocycle_I;
    double ** _cycle_rep;
    double ** _ip_mat;
    double ** rep;
    double ** edge_vectors;
    double ** edge_vectors_T;
    double ** first_product;
    double ** inner_product;
    double ** metric_tensor;
};

PyMODINIT_FUNC init_nloptimize(void)
{
    Py_InitModule("_nloptimize", functions);
    import_array();
    return;
}

static PyObject * nloptimize(PyObject *self, PyObject *args)
{
    int ndim;
    int diag_ind;
    data_info data;
    PyArrayObject* lower_bounds;
    PyArrayObject* upper_bounds;
    PyArrayObject* init_x;
    PyArrayObject* array_x = NULL;
    PyArrayObject* inner_product_matrix;
    PyArrayObject* cycle_rep;
    PyArrayObject* cycle_cocycle_I;
    PyArrayObject* zero_indi, *zero_indj;
    //read in all the parameters
    if (!PyArg_ParseTuple(args, "iiOOOOOOOO",
                          &ndim,
                          &diag_ind,
                          &lower_bounds,
                          &upper_bounds,
                          &init_x,
                          &cycle_rep,
                          &cycle_cocycle_I,
                          &inner_product_matrix,
                          &zero_indi,
                          &zero_indj)){
        return NULL;
    };
    nlopt_opt opt;
    double *x;
    data.lb = get1darrayd(lower_bounds);
    data.ub = get1darrayd(upper_bounds);
    x = get1darrayd(init_x);
    npy_intp* tt;
    tt = PyArray_SHAPE(init_x);
    data.x_size = (int)tt[0];
    data._cycle_cocycle_I = get2darrayd(cycle_cocycle_I);
    data._cycle_rep = get2darrayd(cycle_rep);
    data._ip_mat = get2darrayd(inner_product_matrix);
    data._zi = get1darrayi(zero_indi);
    data._zj = get1darrayi(zero_indj);

    data.ndim = ndim;
    data.diag_ind = diag_ind;
    tt = PyArray_SHAPE(zero_indi);
    data.nz_size = (int)tt[0];
    tt = PyArray_SHAPE(cycle_rep);
    data.cycle_size = (int)tt[0];
    data.rep_size = (int)tt[0] + data.x_size/data.ndim;
    data.rep = construct2darray(data.rep_size, data.ndim);
    tt = PyArray_SHAPE(cycle_cocycle_I);
    data.B_shape = (int) tt[0];
    // B_I * rep = edge_vectors
    // edge_vectors * metric_tensor = first_product
    // first_product * edge_vectors.T = inner_product
    //
    // piecewise calculation of (inner_product[i][j] - _ip_mat[i][j])^2
    // summation of squared errors = return val.
    data.edge_vectors = construct2darray(data.B_shape, data.ndim);
    data.edge_vectors_T = construct2darray(data.ndim, data.B_shape);
    data.first_product = construct2darray(data.B_shape, data.ndim);
    data.inner_product = construct2darray(data.B_shape, data.B_shape);
    data.metric_tensor = construct2darray(data.ndim, data.ndim);

    int res=1;
    data.angle_inds = factorial(ndim, res) / factorial(2, res) / factorial(ndim-2, res); 
    data.start = data.angle_inds + data.ndim;
    /* 
    for (int i=0; i<ndim; i++){
        for (int j=0; j<ndim; j++){
            std::cout<<metric_tensor[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    
    */
    /*
    for (int i=0; i<B_shape; i++){
        for (int j=0; j<ndim; j++){
            std::cout<<edge_vectors[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    

    for (int i=0; i<ndim; i++){
        for (int j=0; j<B_shape; j++){
            std::cout<<edge_vectors_T[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    /*
    tt = PyArray_SHAPE(zero_indi);
    for (int i=0; i<(int)tt[0]; i++){
        std::cout<<_zi[i]<<", "<<_zj[i]<<std::endl;
    }
    */
    //construct a matrix product calculator
    //
    //construct a transpose calculator
    //
    //construct a matrix square difference calculator
    //
    //construct a matrix summation over indices calculator
    //
    //
    //construct the objective function
    opt = nlopt_create(NLOPT_LN_BOBYQA, data.x_size); /* algorithm and dimensionality */
    nlopt_set_lower_bounds(opt, data.lb);
    nlopt_set_upper_bounds(opt, data.ub);
    nlopt_set_min_objective(opt, objectivefunc, &data);
    /*
    unsigned int n;
    double * grad;
    objectivefunc(n, x, grad, &data);
    objectivefunc(n, x, grad, &data);
    */
    nlopt_set_ftol_rel(opt, 1e-8);
    double minf; /* the minimum objective value, upon return */
    //nlopt_optimize(opt, x, &minf);
    
    if (nlopt_optimize(opt, x, &minf) < 0) {
            printf("nlopt failed!\n");
    }
    /*
    else {
            printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    }
    */
    void* xptr;
    npy_intp* dim;
    dim = (npy_intp*) data.x_size;
    PyObject* val;
    array_x = (PyArrayObject*)PyArray_ZEROS(1, dim, NPY_INT, 0);
    /* 
    array_x = (PyArrayObject*) PyArray_ZEROS(1, dim, NPY_FLOAT, 0);
    for (int i=0; i<data.x_size; i++){
        xptr = PyArray_GETPTR1(array_x, i);
        val = PyFloat_FromDouble(x[i]);
        PyArray_SETITEM(array_x, (char*) xptr, val);
        Py_DECREF(val);
    }
    */
    nlopt_destroy(opt); 
    
    free(data.lb);
    free(data.ub);
    free(x);
    free(data._zi);
    free(data._zj);
    free_2d_array(data.edge_vectors, data.B_shape);
    free_2d_array(data.edge_vectors_T, data.ndim);
    free_2d_array(data.first_product, data.B_shape);
    free_2d_array(data.inner_product, data.B_shape);
    free_2d_array(data.metric_tensor, data.ndim);
    free_2d_array(data.rep, data.rep_size);
    free_2d_array(data._cycle_cocycle_I, data.B_shape);
    free_2d_array(data._cycle_rep, data.cycle_size);
    free_2d_array(data._ip_mat, data.B_shape);
    return PyArray_Return(array_x);
}
double objectivefunc(unsigned n, const double *x, double *grad, void *dd)
{
    double ans;
    if (grad) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    data_info d = *((struct data_info *)dd); 
    create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, x, d.rep);
    create_metric_tensor(d.ndim, x, d.metric_tensor);
    matrix_multiply(d.B_shape, d.B_shape, d._cycle_cocycle_I, d.B_shape, d.ndim, d.rep, d.edge_vectors);
    transpose_2darray(d.B_shape, d.ndim, d.edge_vectors, d.edge_vectors_T);
    matrix_multiply(d.B_shape, d.ndim, d.edge_vectors, d.ndim, d.ndim, d.metric_tensor, d.first_product);
    matrix_multiply(d.B_shape, d.ndim, d.first_product, d.ndim, d.B_shape, d.edge_vectors_T, d.inner_product);
    setup_matrix(d.diag_ind, d.B_shape, d.inner_product);
    ans = sumsquarediff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
    //std::cout<<ans<<std::endl;
    /*
    for (int i =0; i<d.nz_size; i++){
        std::cout<<d._zi[i]<<' '<<d._zj[i]<<std::endl;
    }
    */
    return ans; 
}
double ** construct2darray(int rows, int cols){
    double **carray;
    carray = (double**)malloc(sizeof(double*)*rows);
    for (int i=0; i<rows; i++){
        carray[i] = (double*)malloc(sizeof(double)*cols);
    }
    return carray;
}

double * get1darrayd(PyArrayObject* arr){
    PyObject* arr_item;
    void* ind;
    npy_intp * sz = PyArray_SHAPE(arr);
    double *carray;
    double dd;
    int size;
    size = (int) sz[0];
    carray = (double*)malloc(sizeof(double) * size);
    for (int i=0; i<size; i++){
        ind = PyArray_GETPTR1(arr, (npy_intp) i);
        arr_item = PyArray_GETITEM(arr, (char*) ind);
        dd = PyFloat_AsDouble(arr_item);
        carray[i] = dd;
        Py_DECREF(arr_item);
    }
    return carray;
}

int * get1darrayi(PyArrayObject* arr){
    PyObject* arr_item;
    void* ind;
    npy_intp * sz = PyArray_SHAPE(arr);
    int *carray;
    int ii;
    int size;
    size = (int) sz[0];
    carray = (int*)malloc(sizeof(int) * size);
    for (int i=0; i<size; i++){
        ind = PyArray_GETPTR1(arr, (npy_intp) i);
        arr_item = PyArray_GETITEM(arr, (char*) ind);
        ii = (int)PyInt_AsLong(arr_item);
        carray[i] = ii;
        Py_DECREF(arr_item);
    }
    return carray;
}

double ** get2darrayd(PyArrayObject* arr){
    PyObject* arr_item;
    void* ind;
    npy_intp * sz = PyArray_SHAPE(arr);
    double ** carray;
    double dd;
    int size1, size2;
    size1 = (int) sz[0];
    size2 = (int) sz[1];
    carray = (double**)malloc(sizeof(double*)*size1);
    for (int i=0; i<size1; i++){
        carray[i] = (double*)malloc(sizeof(double)*size2);
        for (int j=0; j<size2; j++){
            
            ind = PyArray_GETPTR2(arr, (npy_intp) i, (npy_intp) j);
            arr_item = PyArray_GETITEM(arr, (char*) ind);
            dd = PyFloat_AsDouble(arr_item);
            carray[i][j] = dd;
            Py_DECREF(arr_item);
        }
    }
    return carray;
}

void free_2d_array(double ** carr, int rows){
    for (int i=0; i<rows; i++){
        free(carr[i]);
    }
    free(carr);
    return;
}
    
void create_full_rep(int row1, int col1, double **cycle, int start, int row2, const double *cocycle, double ** rep){
    int counter=start;
    for (int i=0; i<row1+row2; i++){
        if (i<row1){
            for (int j=0; j<col1; j++){
                rep[i][j] = cycle[i][j];
            }
        }   
        else{
            for (int k=0; k<col1; k++){
                rep[i][k] = cocycle[counter];
                counter++;
            }
        }
    }
}

void create_metric_tensor(int ndim, const double *x, double **metric_tensor){
    for (int i=0; i<ndim; i++){
        metric_tensor[i][i] = x[i];
    }
    for (int i=0; i<ndim; i++){
        for (int j=i+1; j<ndim; j++){
            metric_tensor[i][j] = x[ndim+i];
            metric_tensor[j][i] = x[ndim+i];
        }
    }
}

void transpose_2darray(int rows, int cols, double** orig, double** transpose){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            transpose[j][i] = orig[i][j];
        }
    }
}
void matrix_multiply(int row1, int col1, double **mat1, int row2, int col2, double **mat2, double **ans){
    
    double sum;
    if (col1 != row2){
        std::cout<<"Cannot multiply these matrices!!"<<std::endl;
    }
    //generate matrix product array
    for (int c = 0 ; c < row1 ; c++ )
    {
      for (int d = 0 ; d < col2 ; d++ )
      {
        for (int k = 0 ; k < row2 ; k++ )
        {
          sum = sum + mat1[c][k]*mat2[k][d];
        }
 
        ans[c][d] = sum;
        sum = 0;
      }
    }
}

void setup_matrix(int scale_ind, int size, double** inner_prod){
    double ang;
    double sc = inner_prod[scale_ind][scale_ind];
    for (int i=0; i<size; i++){
        for (int j=i+1; j<size; j++){
            ang = inner_prod[i][j] / sqrt(inner_prod[i][i]) / sqrt(inner_prod[j][j]);
            inner_prod[i][j] = ang;
            inner_prod[j][i] = ang;
        }
    }
    for (int i=0; i<size; i++){
        inner_prod[i][i] = inner_prod[i][i] / sc;
    }
}

double sumsquarediff(int size, int* nzi, int* nzj, double** A, double** B){
    double sum=0;
    int m, n;
    for (int i=0; i<size; i++){
        m = nzi[i];
        n = nzj[i];
        
        sum += pow((A[m][n] - B[m][n]), 2);
        //std::cout<<A[m][n]<<std::endl;
    }
    return sum;
}

int factorial(int x, int result = 1) {
      if (x == 1) return result; else return factorial(x - 1, x * result);
}
