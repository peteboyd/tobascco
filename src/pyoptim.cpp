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
void forward_difference_grad(double*, const double*, double, void*, double);
void central_difference_grad(double*, const double* , void*, double);
double objectivefunc(unsigned, const double*, double*, void*);
void create_metric_tensor(int, const double*, double*);
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
    int *_zi, *_zj;
    double ** _cycle_cocycle_I;
    double ** _cycle_rep;
    double ** _ip_mat;
    double ** rep;
    double ** edge_vectors;
    double ** edge_vectors_T;
    double ** M1;
    double * Z;
    double * diag;
    double * diag2;
    double * farray;
    double * barray;
};

double compute_inner_product_fast(const double*, data_info);

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
    double xrel; /* the relative tolerance for the input variables */
    double frel; /* the relative tolerance for the function change */
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
    if (!PyArg_ParseTuple(args, "iiOOOOOOOOdd",
                          &ndim,
                          &diag_ind,
                          &lower_bounds,
                          &upper_bounds,
                          &init_x,
                          &cycle_rep,
                          &cycle_cocycle_I,
                          &inner_product_matrix,
                          &zero_indi,
                          &zero_indj,
                          &xrel,
                          &frel)){
        return NULL;
    };
    nlopt_opt opt;
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
    data.Z = (double*)malloc(sizeof(double) * 6);
    data.M1 = construct2darray(data.B_shape, data.ndim);
    data.farray = (double*)malloc(sizeof(double) * data.x_size);
    data.barray = (double*)malloc(sizeof(double) * data.x_size);
    data.diag = (double*)malloc(sizeof(double) * data.B_shape);
    data.diag2 = (double*)malloc(sizeof(double) * data.B_shape);
    int res=1;
    data.angle_inds = factorial(ndim, res) / factorial(2, res) / factorial(ndim-2, res); 
    data.start = data.angle_inds + data.ndim;
    
   
    //LOCAL OPTIMIZER***********************************
    opt = nlopt_create(NLOPT_LD_LBFGS, data.x_size);
    nlopt_set_vector_storage(opt, 10000); /* for quasi-newton algorithms, how many gradients to store */
    nlopt_set_min_objective(opt, objectivefunc, &data);
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    nlopt_set_xtol_rel(opt, xrel);
    retval = nlopt_optimize(opt, x, &minf);
    if (retval < 0) {
            printf("nlopt failed!\n");
            Py_INCREF(Py_None);
            return(Py_None);
    }
    //END LOCAL OPTIMIZER******************************* 
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
    free_2d_array(data.rep, data.rep_size);
    free_2d_array(data._cycle_cocycle_I, data.B_shape);
    free_2d_array(data._cycle_rep, data.cycle_size);
    free_2d_array(data._ip_mat, data.B_shape);
    free_2d_array(data.M1, data.B_shape);
    free(data.farray);
    free(data.barray);
    free(data.diag);
    free(data.diag2);
    free(data.Z);
    return PyArray_Return(array_x); 
}


double objectivefunc(unsigned n, const double *x, double *grad, void *dd)
{
    double ans;
    data_info d = *((struct data_info *)dd); 
    create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, x, d.rep);
    create_metric_tensor(d.ndim, x, d.Z);
    ans = compute_inner_product_fast(x, d);
    if (grad) {
        //forward_difference_grad(grad, x, ans, dd, 1e-5);
        central_difference_grad(grad, x, dd, 1e-4);
    }
    //std::cout<<ans<<std::endl;
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

void create_metric_tensor(int ndim, const double *x, double *Z){
    //only for ndim = 3!!
    Z[0] = x[0];
    Z[1] = x[1];
    Z[2] = x[2];
    Z[3] = x[3]*sqrt(x[0])*sqrt(x[1]);
    Z[4] = x[4]*sqrt(x[0])*sqrt(x[2]);
    Z[5] = x[5]*sqrt(x[1])*sqrt(x[2]);
}

void forward_difference_grad(double* grad, const double* x, double fval, void* data, double xinc){
    data_info d = *((struct data_info *)data);
    double ans;
    for (int i=0; i<d.x_size; i++){
        memcpy((void*)d.farray, (void*)x, d.x_size*sizeof(double));
        d.farray[i] += xinc;
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, d.farray, d.rep);
        create_metric_tensor(d.ndim, d.farray, d.Z);
        ans = compute_inner_product_fast(d.farray, d);
        grad[i] = (ans - fval)/xinc;
    }
}

void central_difference_grad(double* grad, const double* x, void* data, double xinc){
    data_info d = *((struct data_info *)data);
    double forward, back;
    for (int i=0; i<d.x_size; i++){
        memcpy((void*)d.farray, (void*)x, d.x_size*sizeof(double));
        memcpy((void*)d.barray, (void*)x, d.x_size*sizeof(double));
        d.farray[i] += xinc;
        d.barray[i] -= xinc;
        //forward grad
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, d.farray, d.rep);
        create_metric_tensor(d.ndim, d.farray, d.Z);
        forward = compute_inner_product_fast(d.farray, d);

        //backward grad
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, d.barray, d.rep);
        create_metric_tensor(d.ndim, d.barray, d.Z);
        back = compute_inner_product_fast(d.barray,d);
        grad[i] = (forward - back)/(2.*xinc);
    }
}
double compute_inner_product_fast(const double *x, data_info d){
    //Try to eliminate some inefficiencies in the code, no more explicit transposing
    //Optimize matrix products to reduce the number of multiplications all in a singls
    //function.
    //
    //data_info d = *((struct data_info *)data);
    double max,sum,squarediff;
    int i, j, t, s;
    //row1, col2, row2
    //B*-1 * alpha(B) 
    //resulting B_shape * ndim array
    sum=0;
    squarediff=0;

    for (int i = 0 ; i < d.B_shape ; i++ ){
        for (int j = 0 ; j < d.ndim ; j++ ){
            for (int k = 0 ; k < d.B_shape ; k++ ){
                sum += d._cycle_cocycle_I[i][k]*d.rep[k][j];
            }
        
            d.M1[i][j] = sum;
            sum=0; 
 
        }
    }
    

    sum=0;
    //need the diagonal values first
    for (int r = 0; r < d.B_shape; r++){
        for (int k = 0; k < d.ndim; k++){
            // VALID ONLY FOR d.ndim == 3!!!!!!!!!
            t = (k*2)%d.ndim;
            s = (k*2+1)%d.ndim;
            sum += d.Z[k] * d.M1[r][k] * d.M1[r][k];
            sum += d.Z[k+3] * 2 * d.M1[r][t]*d.M1[r][s];
        }
        if(sum<0)sum=500;
        d.diag2[r] = sum;
        d.diag[r] = sqrt(sum);

        sum=0;
    }
    max=d.diag2[d.diag_ind];

    for (int r = 0 ; r < d.nz_size ; r++ ){
        i = d._zi[r];
        j = d._zj[r];
        if(i==j){
            squarediff += pow((d.diag2[i]/max - d._ip_mat[i][j]), 2);
        }
        else{
            for (int k = 0 ; k < d.ndim ; k++ ){
                sum += d.Z[k] * (d.M1[i][k]*d.M1[j][k]);
            
                // VALID ONLY FOR d.ndim == 3!!!!!!!!!
                t = (k*2)%d.ndim;
                s = (k*2+1)%d.ndim;

                sum += d.Z[k+3]*(d.M1[i][t]*d.M1[j][s] + d.M1[i][s]*d.M1[j][t]);
            }

            squarediff += pow((sum/d.diag[i]/d.diag[j] - d._ip_mat[i][j]), 2);
            sum=0;
        }
    }
    return squarediff;
}

int factorial(int x, int result = 1) {
    if (x == 1) return result; else return factorial(x - 1, x * result);
}
