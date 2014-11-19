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
static PyObject * get_ip_matrix(PyObject *self, PyObject *args);
static PyObject * matrix_product(PyObject *self, PyObject *args);
static PyObject * triple_matrix_product(PyObject *self, PyObject *args);
void forward_difference_grad(double*, const double*, double, void*, double);
void central_difference_grad(double*, const double* , void*, double);
double objectivefunc(unsigned, const double*, double*, void*);
double sumsquarediff(int, int*, int*, double**, double**);
double sumabsdiff(int, int*, int*, double**, double**);
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
    {"c_inner_prod", get_ip_matrix, METH_VARARGS, NULL},
    {"c_matrix_prod", matrix_product, METH_VARARGS, NULL},
    {"c_trip_prod", triple_matrix_product, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL} 
};


struct data_info{
    int rep_size, cycle_size, nz_size, x_size;
    int B_shape, ndim, diag_ind;
    int angle_inds, start;
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
    double minf; /* the minimum objective value, upon return */
    data_info data;
    nlopt_result retval; /*Return value from nlopt: 1 = GENERAL SUCCESS
                                                    2 = STOPVAL REACHED
                                                    3 = FTOL REACHED
                                                    4 = XTOL REACHED
                                                    5 MAXEVAL REACHED
                                                    6 = MAXTIME REACHED*/
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
    nlopt_opt opt, local_opt;
    double *x, *lb, *ub;
    lb = get1darrayd(lower_bounds);
    ub = get1darrayd(upper_bounds);
    x = get1darrayd(init_x);
    npy_intp* tt;
    tt = PyArray_DIMS(init_x);
    data.x_size = (int)tt[0];
    data._cycle_cocycle_I = get2darrayd(cycle_cocycle_I);
    data._cycle_rep = get2darrayd(cycle_rep);
    data._ip_mat = get2darrayd(inner_product_matrix);
    data._zi = get1darrayi(zero_indi);
    data._zj = get1darrayi(zero_indj);
    
    //PyObject_Print((PyObject*)zero_indj, stdout, 0);
    data.ndim = ndim;
    data.diag_ind = diag_ind;
    tt = PyArray_DIMS(zero_indi);
    data.nz_size = (int)tt[0];
    tt = PyArray_DIMS(cycle_rep);
    data.cycle_size = (int)tt[0];
    data.rep_size = (int)tt[0] + data.x_size/data.ndim;
    data.rep = construct2darray(data.rep_size, data.ndim);
    tt = PyArray_DIMS(cycle_cocycle_I);
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
    for (int i=0; i<data.B_shape; i++){
        for (int j=0; j<data.B_shape; j++){
            std::cout<<data._cycle_cocycle_I[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    /* 
    for (int i=0; i<data.ndim; i++){
        for (int j=0; j<data.ndim; j++){
            std::cout<<data.metric_tensor[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    
    
   
    for (int i=0; i<data.B_shape; i++){
        for (int j=0; j<data.ndim; j++){
            std::cout<<data.edge_vectors[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    

    for (int i=0; i<data.ndim; i++){
        for (int j=0; j<data.B_shape; j++){
            std::cout<<data.edge_vectors_T[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    /*
    tt = PyArray_DIMS(zero_indi);
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
    //opt = nlopt_create(NLOPT_LD_TNEWTON, data.x_size); /* algorithm and dimensionality */
    //opt = nlopt_create(NLOPT_LN_COBYLA, data.x_size);

    //GLOBAL OPTIMIZER**********************************
    //opt = nlopt_create(NLOPT_GN_ESCH, data.x_size);
    opt = nlopt_create(NLOPT_GN_DIRECT_L, data.x_size);
    nlopt_set_min_objective(opt, objectivefunc, &data);
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    //nlopt_set_ftol_abs(opt, 0.01);
    nlopt_set_ftol_rel(opt, 1e-10);

    //nlopt_set_population(opt, 10);
    ////MLSL specific*************************************
    //local_opt = nlopt_create(NLOPT_LD_LBFGS, data.x_size);
    //nlopt_set_ftol_rel(local_opt, 1e-5);
    //nlopt_set_local_optimizer(opt, local_opt);
    //nlopt_destroy(local_opt);

    //retval = nlopt_optimize(opt, x, &minf);
    //if (retval < 0) {
    //        printf("nlopt failed!\n");
    //        Py_INCREF(Py_None);
    //        return(Py_None);
    //}
    //std::cout<<retval<<std::endl;
    nlopt_destroy(opt); 

    //END GLOBAL OPTIMIZER******************************

    //LOCAL OPTIMIZER***********************************
    opt = nlopt_create(NLOPT_LD_LBFGS, data.x_size);
    //nlopt_set_initial_step1(opt, 0.1);
    nlopt_set_vector_storage(opt, 10000); /* for quasi-newton algorithms, how many gradients to store */
    /*
    for (int i=0; i<data.x_size; i++){
        std::cout<<ub[i]<<" ";
    }
    std::cout<<std::endl;
    */
    //ub[0] = HUGE_VAL;
    //ub[1] = HUGE_VAL;
    //ub[2] = HUGE_VAL;
    nlopt_set_min_objective(opt, objectivefunc, &data);
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    
    //nlopt_set_ftol_rel(opt, 1e-10);
    nlopt_set_xtol_rel(opt, 1e-5);
    retval = nlopt_optimize(opt, x, &minf);
    if (retval < 0) {
            printf("nlopt failed!\n");
            Py_INCREF(Py_None);
            return(Py_None);
    }
    //END LOCAL OPTIMIZER******************************* 
    //std::cout<<retval<<std::endl;
    /*
    else {
            printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    }
    */
    void* xptr;
    npy_intp dim = data.x_size;
    PyObject* val; 
    array_x = (PyArrayObject*) PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
     
    for (int i=0; i<data.x_size; i++){
        xptr = PyArray_GETPTR1(array_x, i);
        val = PyFloat_FromDouble(x[i]);
        PyArray_SETITEM(array_x, (char*) xptr, val);
        Py_DECREF(val);
    }
    
    nlopt_destroy(opt); 
    free(lb);
    free(ub);
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

static PyObject * get_ip_matrix(PyObject *self, PyObject *args)
{
    int ndim;
    data_info d;
    PyArrayObject* init_x;
    PyArrayObject* array_x = NULL;
    PyArrayObject* inner_product_matrix;
    PyArrayObject* cycle_rep;
    PyArrayObject* cycle_cocycle_I;
    //read in all the parameters
    if (!PyArg_ParseTuple(args, "iOOOO",
                          &ndim,
                          &init_x,
                          &cycle_rep,
                          &cycle_cocycle_I,
                          &inner_product_matrix)){
        return NULL;
    };

    double *x;
    x = get1darrayd(init_x);
    npy_intp* tt;
    tt = PyArray_DIMS(init_x);
    d.x_size = (int)tt[0];
    d.start = 0;
    d.diag_ind = 0;
    d._cycle_cocycle_I = get2darrayd(cycle_cocycle_I);
    d._cycle_rep = get2darrayd(cycle_rep);
    d._ip_mat = get2darrayd(inner_product_matrix);
    
    d.ndim = ndim;
    tt = PyArray_DIMS(cycle_rep);
    d.cycle_size = (int)tt[0];
    d.rep_size = (int)tt[0] + d.x_size/d.ndim;
    d.rep = construct2darray(d.rep_size, d.ndim);
    tt = PyArray_DIMS(cycle_cocycle_I);
    d.B_shape = (int) tt[0];
    d.edge_vectors = construct2darray(d.B_shape, d.ndim);
    d.edge_vectors_T = construct2darray(d.ndim, d.B_shape);
    d.first_product = construct2darray(d.B_shape, d.ndim);
    d.inner_product = construct2darray(d.B_shape, d.B_shape);
    d.metric_tensor = construct2darray(d.ndim, d.ndim);

    create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, x, d.rep);
    int xinc =6;
    for (int i=d.cycle_size; i<d.B_shape; i++){
        for (int j=0; j<d.ndim; j++){
            d.rep[i][j] = x[xinc];
            xinc ++;
        }
    }
    create_metric_tensor(d.ndim, x, d.metric_tensor);
    matrix_multiply(d.B_shape, d.B_shape, d._cycle_cocycle_I, d.B_shape, d.ndim, d.rep, d.edge_vectors);
    matrix_multiply(d.B_shape, d.ndim, d.edge_vectors, d.ndim, d.ndim, d.metric_tensor, d.first_product);
    transpose_2darray(d.B_shape, d.ndim, d.edge_vectors, d.edge_vectors_T);
    /*
    for (int i=0; i<d.ndim; i++){
        for (int j=0; j<d.ndim; j++){
            std::cout<<d.metric_tensor[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */ 
    matrix_multiply(d.B_shape, d.ndim, d.first_product, d.ndim, d.B_shape, d.edge_vectors_T, d.inner_product);
    /* 
    for (int i=0; i<d.B_shape; i++){
        for (int j=0; j<d.B_shape; j++){
            std::cout<<d.inner_product[i][j]<<" ";
        }
        std::cout<<std::endl;
        //std::cout<<d.inner_product[i][i]<<" ";
    }
    */
    setup_matrix(4, d.B_shape, d.inner_product);
    
    npy_intp dims[2] = {(npy_intp)d.B_shape, (npy_intp)d.B_shape};
    array_x = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    void* xptr;
    PyObject* val; 
    for (int i=0; i<d.B_shape; i++){
        for (int j=0; j<d.B_shape; j++){
            xptr = PyArray_GETPTR2(array_x, i, j);
            val = PyFloat_FromDouble(d.inner_product[i][j]);
            PyArray_SETITEM(array_x, (char*) xptr, val);
            Py_DECREF(val);
        }
    }
    
    free(x);
    free_2d_array(d.edge_vectors, d.B_shape);
    free_2d_array(d.edge_vectors_T, d.ndim);
    free_2d_array(d.first_product, d.B_shape);
    free_2d_array(d.inner_product, d.B_shape);
    free_2d_array(d.metric_tensor, d.ndim);
    free_2d_array(d.rep, d.rep_size);
    free_2d_array(d._cycle_cocycle_I, d.B_shape);
    free_2d_array(d._cycle_rep, d.cycle_size);
    free_2d_array(d._ip_mat, d.B_shape);
    return PyArray_Return(array_x); 
}

static PyObject * triple_matrix_product(PyObject *self, PyObject *args)
{
    int n1,m1,n2,m2,n3,m3;
    data_info d;
    npy_intp * tt;
    PyArrayObject* array_x = NULL;
    PyArrayObject* mat1;
    PyArrayObject* mat2;
    PyArrayObject* mat3;
    //read in all the parameters
    if (!PyArg_ParseTuple(args, "OOO",
                          &mat1,
                          &mat2,
                          &mat3)){
        return NULL;
    };

    d._cycle_cocycle_I = get2darrayd(mat1);
    d._cycle_rep = get2darrayd(mat2);
    d.inner_product = get2darrayd(mat3);

    tt = PyArray_DIMS(mat1);
    n1 = (int)tt[0];
    m1 = (int)tt[1];
    tt = PyArray_DIMS(mat2);
    n2 = (int)tt[0];
    m2 = (int)tt[1];
    tt = PyArray_DIMS(mat3);
    n3 = (int)tt[0];
    m3 = (int)tt[1];
    d.edge_vectors = construct2darray(n1, m2);
    matrix_multiply(n1, m1, d._cycle_cocycle_I, n2, m2, d._cycle_rep, d.edge_vectors);
    d.first_product = construct2darray(n1,m3);
    matrix_multiply(n1, m2, d.edge_vectors, n3, m3, d.inner_product, d.first_product);
    /*
    for (int i=0; i<n1; i++){
        for (int j=0; j<m3; j++){
            std::cout<<d.first_product[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    npy_intp dims[2] = {(npy_intp)n1, (npy_intp)m3};
    array_x = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    void* xptr;
    PyObject* val;
    double max =0 ; 
    for (int i=0; i<n1; i++){
        for (int j=0; j<m3; j++){
            xptr = PyArray_GETPTR2(array_x, i, j);
            val = PyFloat_FromDouble(d.first_product[i][j]);
            if (d.first_product[i][j] > max){
                max = d.first_product[i][j];
            }

            PyArray_SETITEM(array_x, (char*) xptr, val);
            Py_DECREF(val);
        }
    }
    free_2d_array(d._cycle_cocycle_I, n1);
    free_2d_array(d._cycle_rep, n2);
    free_2d_array(d.inner_product, n3);
    free_2d_array(d.edge_vectors, n1);
    free_2d_array(d.first_product, n1);

    return PyArray_Return(array_x); 
}
static PyObject * matrix_product(PyObject *self, PyObject *args)
{
    int n1,m1,n2,m2;
    data_info d;
    npy_intp * tt;
    PyArrayObject* array_x = NULL;
    PyArrayObject* mat1;
    PyArrayObject* mat2;
    //read in all the parameters
    if (!PyArg_ParseTuple(args, "OO",
                          &mat1,
                          &mat2)){
        return NULL;
    };

    d._cycle_cocycle_I = get2darrayd(mat1);
    d._cycle_rep = get2darrayd(mat2);
    tt = PyArray_DIMS(mat1);
    n1 = (int)tt[0];
    m1 = (int)tt[1];
    tt = PyArray_DIMS(mat2);
    n2 = (int)tt[0];
    m2 = (int)tt[1];
    d.edge_vectors = construct2darray(n1, m2);
    matrix_multiply(n1, m1, d._cycle_cocycle_I, n2, m2, d._cycle_rep, d.edge_vectors);
    
    npy_intp dims[2] = {(npy_intp)n1, (npy_intp)m2};
    array_x = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    void* xptr;
    PyObject* val; 
    for (int i=0; i<n1; i++){
        for (int j=0; j<m2; j++){
            xptr = PyArray_GETPTR2(array_x, i, j);
            val = PyFloat_FromDouble(d.edge_vectors[i][j]);
            PyArray_SETITEM(array_x, (char*) xptr, val);
            Py_DECREF(val);
        }
    }
    
    free_2d_array(d.edge_vectors, n1);
    free_2d_array(d._cycle_cocycle_I, n1);
    free_2d_array(d._cycle_rep, n2);
    return PyArray_Return(array_x); 
}

double objectivefunc(unsigned n, const double *x, double *grad, void *dd)
{
    double ans;
    data_info d = *((struct data_info *)dd); 
    create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, x, d.rep);
    create_metric_tensor(d.ndim, x, d.metric_tensor);
    matrix_multiply(d.B_shape, d.B_shape, d._cycle_cocycle_I, d.B_shape, d.ndim, d.rep, d.edge_vectors);
    transpose_2darray(d.B_shape, d.ndim, d.edge_vectors, d.edge_vectors_T);
    matrix_multiply(d.B_shape, d.ndim, d.edge_vectors, d.ndim, d.ndim, d.metric_tensor, d.first_product);
    matrix_multiply(d.B_shape, d.ndim, d.first_product, d.ndim, d.B_shape, d.edge_vectors_T, d.inner_product);
    setup_matrix(d.diag_ind, d.B_shape, d.inner_product);
    /*
    for (int i=0; i<d.B_shape; i++){
        for (int j=0; j<d.B_shape; j++){
            std::cout<<d.inner_product[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    exit(0);
    */
    ans = sumsquarediff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
    //ans = sumabsdiff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
    if (grad) {
        //;
        //forward_difference_grad(grad, x, ans, dd, 1e-3);
        //std::cout<<"HERE"<<std::endl;
        central_difference_grad(grad, x, dd, 1e-12);
    }
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
        for (int j=0; j<cols; j++){
            carray[i][j] = 0.0;
        }
    }
    return carray;
}

double * get1darrayd(PyArrayObject* arr){
    PyObject* arr_item;
    void* ind;
    npy_intp * sz = PyArray_DIMS(arr);
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
    npy_intp * sz = PyArray_DIMS(arr);
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
    npy_intp * sz = PyArray_DIMS(arr);
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
    double ijval;
    int counter=0;
    for (int i=0; i<ndim; i++){
        for (int j=i+1; j<ndim; j++){
            ijval = x[ndim+counter]* sqrt(metric_tensor[i][i]) * sqrt(metric_tensor[j][j]);
            metric_tensor[i][j] = ijval; 
            metric_tensor[j][i] = ijval;
            counter++;
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
    
    double sum=0;
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



void forward_difference_grad(double* grad, const double* x, double fval, void* data, double xinc){
    data_info d = *((struct data_info *)data);
    double ans;
    double* tempcopy;
    tempcopy = (double*)malloc(sizeof(double) * d.x_size);
    for (int i=0; i<d.x_size; i++){
        memcpy((void*)tempcopy, (void*)x, d.x_size*sizeof(double));
        tempcopy[i] += xinc;
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, tempcopy, d.rep);
        create_metric_tensor(d.ndim, tempcopy, d.metric_tensor);
        matrix_multiply(d.B_shape, d.B_shape, d._cycle_cocycle_I, d.B_shape, d.ndim, d.rep, d.edge_vectors);
        transpose_2darray(d.B_shape, d.ndim, d.edge_vectors, d.edge_vectors_T);
        matrix_multiply(d.B_shape, d.ndim, d.edge_vectors, d.ndim, d.ndim, d.metric_tensor, d.first_product);
        matrix_multiply(d.B_shape, d.ndim, d.first_product, d.ndim, d.B_shape, d.edge_vectors_T, d.inner_product);
        setup_matrix(d.diag_ind, d.B_shape, d.inner_product);
        ans = sumsquarediff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
        //ans = sumabsdiff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
        grad[i] = (ans - fval)/xinc;
    }
    free(tempcopy);
}

void central_difference_grad(double* grad, const double* x, void* data, double xinc){
    data_info d = *((struct data_info *)data);
    double forward, back;
    double *farray, *barray;
    farray = (double*)malloc(sizeof(double) * d.x_size);
    barray = (double*)malloc(sizeof(double) * d.x_size);
    for (int i=0; i<d.x_size; i++){
        memcpy((void*)farray, (void*)x, d.x_size*sizeof(double));
        memcpy((void*)barray, (void*)x, d.x_size*sizeof(double));
        farray[i] += xinc;
        barray[i] -= xinc;
        //forward grad
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, farray, d.rep);
        create_metric_tensor(d.ndim, farray, d.metric_tensor);
        matrix_multiply(d.B_shape, d.B_shape, d._cycle_cocycle_I, d.B_shape, d.ndim, d.rep, d.edge_vectors);
        transpose_2darray(d.B_shape, d.ndim, d.edge_vectors, d.edge_vectors_T);
        matrix_multiply(d.B_shape, d.ndim, d.edge_vectors, d.ndim, d.ndim, d.metric_tensor, d.first_product);
        matrix_multiply(d.B_shape, d.ndim, d.first_product, d.ndim, d.B_shape, d.edge_vectors_T, d.inner_product);
        setup_matrix(d.diag_ind, d.B_shape, d.inner_product);
        forward = sumsquarediff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
        //forward = sumabsdiff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);

        //backward grad
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, barray, d.rep);
        create_metric_tensor(d.ndim, barray, d.metric_tensor);
        matrix_multiply(d.B_shape, d.B_shape, d._cycle_cocycle_I, d.B_shape, d.ndim, d.rep, d.edge_vectors);
        transpose_2darray(d.B_shape, d.ndim, d.edge_vectors, d.edge_vectors_T);
        matrix_multiply(d.B_shape, d.ndim, d.edge_vectors, d.ndim, d.ndim, d.metric_tensor, d.first_product);
        matrix_multiply(d.B_shape, d.ndim, d.first_product, d.ndim, d.B_shape, d.edge_vectors_T, d.inner_product);
        setup_matrix(d.diag_ind, d.B_shape, d.inner_product);
        back = sumsquarediff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);
        //back = sumabsdiff(d.nz_size, d._zi, d._zj, d.inner_product, d._ip_mat);

        grad[i] = (forward - back)/(2.*xinc);
    }
    free(farray);
    free(barray);
}

void setup_matrix(int scale_ind, int size, double** inner_prod){
    double ang;
    double max = inner_prod[scale_ind][scale_ind];
    //double max = 0;
    for (int i=0; i<size; i++){
        for (int j=i+1; j<size; j++){
            if (inner_prod[i][i] < 0.0)inner_prod[i][i] = 500.0;
            if (inner_prod[j][j] < 0.0)inner_prod[j][j] = 500.0; //penalize for negative lengths.
            //std::cout<<inner_prod[i][i]<<" "<<inner_prod[j][j]<<std::endl;
            ang = inner_prod[i][j] / sqrt(inner_prod[i][i]) / sqrt(inner_prod[j][j]);
            //std::cout<<i<<" "<<inner_prod[i][j]<<std::endl;

            inner_prod[i][j] = ang;
            inner_prod[j][i] = ang;
        }
    }
    /*
    for (int i=0; i<size; i++){
        if (max < inner_prod[i][i]){
            max = inner_prod[i][i];
        }
    }
    */
    for (int i=0; i<size; i++){
        inner_prod[i][i] = inner_prod[i][i] / max;
    }
   
}

double sumabsdiff(int size, int* nzi, int* nzj, double** A, double** B){
    double sum=0;
    int m, n;
    for (int i=0; i<size; i++){
        m = nzi[i];
        n = nzj[i];
        
        sum += fabs(A[m][n] - B[m][n]);
        //std::cout<<A[m][n]<<std::endl;
        //std::cout<<sum<<std::endl;
    }
    return sum;
}

double sumsquarediff(int size, int* nzi, int* nzj, double** A, double** B){
    double sum=0;
    double diff;
    int m, n;
    //std::cout<<"np.array([";
    for (int i=0; i<size; i++){
        m = nzi[i];
        n = nzj[i];
        //std::cout<<pow(A[m][n] - B[m][n], 2)<<", ";
        //std::cout<<A[m][n]<<" "<<m<<" "<<n<<std::endl;
        diff = pow((A[m][n] - B[m][n]), 2);
        //weight the distances more.
        //if( m != n ) diff = 8*diff;
        //if( m == n ) diff = 19*diff;
        sum += diff;
        //std::cout<<A[m][n]<<std::endl;
    }
    //std::cout<<"0.])"<<std::endl;
    //std::cout<<sum<<std::endl;
    return sqrt(sum);
}

int factorial(int x, int result = 1) {
      if (x == 1) return result; else return factorial(x - 1, x * result);
}
