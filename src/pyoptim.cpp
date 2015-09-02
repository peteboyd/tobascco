#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string.h>
#include <math.h>
#include <nlopt.h>
#include <numpy/arrayobject.h>

void create_full_rep(int, int, double**, int, int, const double*, double**);
static PyObject * nloptimize(PyObject *self, PyObject *args);
void forward_difference_grad(double*, const double*, double, void*, double);
void central_difference_grad(double*, const double* , void*, double);
double objectivefunc(unsigned, const double*, double*, void*);
double objectivefunc2D(unsigned, const double*, double*, void*);
void central_difference_grad2D(double*, const double* , void*, double);
void create_metric_tensor(int, const double*, double*);
void create_metric_tensor2D(int, const double*, double*);
double * get1darrayd(PyArrayObject*);
int * get1darrayi(PyArrayObject*);
double ** get2darrayd(PyArrayObject*);
double ** construct2darray(int rows, int cols);
void free_2d_array(double**, int);

static PyMethodDef functions[] = {
    {"nloptimize", nloptimize, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL} 
};


struct data_info{
    int rep_size, cycle_size, nz_size, x_size;
    int B_shape, ndim, diag_ind;
    int start;
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
    double * stored_dp; 
};

double compute_inner_product_fast(const double*, data_info);
double compute_inner_product_fast2D(const double*, data_info);
void jacobian3D_sums(double*, const double* , data_info);

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
                                                    5 = MAXEVAL REACHED
                                                    6 = MAXTIME REACHED*/
    nlopt_algorithm global; //global optimizer
    nlopt_algorithm local; //local optimizer
    PyArrayObject* lower_bounds;
    PyArrayObject* upper_bounds;
    PyArrayObject* init_x;
    PyArrayObject* array_x = NULL;
    PyArrayObject* inner_product_matrix;
    PyArrayObject* cycle_rep;
    PyArrayObject* cycle_cocycle_I;
    PyArrayObject* zero_indi, *zero_indj;
    PyObject* pgoptim, *ploptim;
    //read in all the parameters
    if (!PyArg_ParseTuple(args, "iiOOOOOOOOddOO",
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
                          &frel,
                          &pgoptim,
                          &ploptim)){
        return NULL;
    };
    nlopt_opt opt, local_opt;
    double *x, *lb, *ub;
    std::string goptim=PyString_AsString(pgoptim);
    std::string loptim=PyString_AsString(ploptim);
    
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
    if(ndim == 3){
        data.Z = (double*)malloc(sizeof(double) * 6);
        data.start = 6; 
    }
    else if(data.ndim == 2){
        data.Z = (double*)malloc(sizeof(double) * 3);
        data.start = 3; 
    }
    data.M1 = construct2darray(data.B_shape, data.ndim);
    data.farray = (double*)malloc(sizeof(double) * data.x_size);
    data.barray = (double*)malloc(sizeof(double) * data.x_size);
    data.diag = (double*)malloc(sizeof(double) * data.B_shape);
    data.diag2 = (double*)malloc(sizeof(double) * data.B_shape);
    data.stored_dp = (double*)malloc(sizeof(double) * data.nz_size);
    //initialize the local and global optimizers, so the compilation doesn't spew
    // out useless warnings
    global=NLOPT_GN_DIRECT;
    local=NLOPT_LD_LBFGS;
    //determine local optimizer
    if (loptim == "cobyla")local=NLOPT_LN_COBYLA;
    else if (loptim == "bobyqa")local=NLOPT_LN_BOBYQA;
    else if (loptim == "newoua")local=NLOPT_LN_NEWUOA_BOUND;
    else if (loptim == "praxis")local=NLOPT_LN_PRAXIS;
    else if (loptim == "nelder-mead")local=NLOPT_LN_NELDERMEAD;
    else if (loptim == "mma")local=NLOPT_LD_MMA;
    else if (loptim == "ccsa")local=NLOPT_LD_CCSAQ;
    else if (loptim == "slsqp")local=NLOPT_LD_SLSQP;
    else if (loptim == "lbfgs")local=NLOPT_LD_LBFGS;
    else if (loptim == "newton")local=NLOPT_LD_TNEWTON;
    else if (loptim == "newton-restart")local=NLOPT_LD_TNEWTON_RESTART;
    else if (loptim == "newton-precond")local=NLOPT_LD_TNEWTON_PRECOND;
    else if (loptim == "newton-precond-restart")local=NLOPT_LD_TNEWTON_PRECOND_RESTART;
    else if (loptim == "var1")local=NLOPT_LD_VAR1;
    else if (loptim == "var2")local=NLOPT_LD_VAR2;

    //GLOBAL OPTIMIZER***********************************
    if (!goptim.empty()){
        if (goptim == "direct")global=NLOPT_GN_DIRECT;
        else if (goptim == "direct")global=NLOPT_GN_DIRECT;
        else if (goptim == "direct-l")global=NLOPT_GN_DIRECT_L;
        //else if (goptim == "direct-l-rand")global=NLOPT_GLOBAL_DIRECT_L_RAND;
        //else if (goptim == "direct-noscale")global=NLOPT_GLOBAL_DIRECT_NOSCAL;
        //else if (goptim == "direct-l-noscale")global=NLOPT_GLOBAL_DIRECT_L_NOSCAL;
        //else if (goptim == "direct-l-rand-noscale")global=NLOPT_GLOBAL_DIRECT_L_RAND_NOSCAL;
        else if (goptim == "crs2")global=NLOPT_GN_CRS2_LM;
        else if (goptim == "stogo")global=NLOPT_GD_STOGO;
        else if (goptim == "stogo-rand")global=NLOPT_GD_STOGO_RAND;
        else if (goptim == "isres")global=NLOPT_GN_ISRES;
        else if (goptim == "esch")global=NLOPT_GN_ESCH;
        else if (goptim == "mlsl")global=NLOPT_G_MLSL;
        else if (goptim == "mlsl-lds")global=NLOPT_G_MLSL_LDS;
        opt = nlopt_create(global, data.x_size);
        // create local optimizer for the mlsl algorithms.
        if ((goptim == "mlsl") || (goptim == "mlsl-lds")){
            local_opt = nlopt_create(local, data.x_size);
            nlopt_set_local_optimizer(opt, local_opt);
        }
        if(ndim==3){
            nlopt_set_min_objective(opt, objectivefunc, &data);
        }
        else if(ndim==2){
            nlopt_set_min_objective(opt, objectivefunc2D, &data);
        }

        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_xtol_rel(opt, xrel);  // set absolute tolerance on the change in the input parameters
        nlopt_set_ftol_abs(opt, frel);  // set absolute tolerance on the change in the objective funtion
        retval = nlopt_optimize(opt, x, &minf);
        if (retval < 0) {
                printf("global nlopt failed!\n");
        }
        nlopt_destroy(opt); 
    }
    /*else{
        printf("No global optimisation requested, preparing local optimiser\n");
    }*/
    //END GLOBAL OPTIMIZER******************************* 
   
    //LOCAL OPTIMIZER***********************************
    opt = nlopt_create(local, data.x_size);
    nlopt_set_vector_storage(opt, 10000); // for quasi-newton algorithms, how many gradients to store 
    if(ndim==3){
        nlopt_set_min_objective(opt, objectivefunc, &data);
    }
    else if(ndim==2){
        nlopt_set_min_objective(opt, objectivefunc2D, &data);
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    nlopt_set_xtol_rel(opt, xrel);  // set absolute tolerance on the change in the input parameters
    nlopt_set_ftol_abs(opt, frel);  // set absolute tolerance on the change in the objective funtion
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
    free(data.stored_dp);
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
        //Jacobian calc not working!!
        //jacobian3D_sums(grad, x, d);
        //forward_difference_grad(grad, x, ans, dd, 1e-5);
        central_difference_grad(grad, x, dd, 1e-5);
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

void create_metric_tensor2D(int ndim, const double *x, double *Z){
    //only for ndim = 3!!
    Z[0] = x[0];
    Z[1] = x[1];
    Z[2] = x[2]*sqrt(x[0])*sqrt(x[1]);
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
        //std::cout<<grad[i]<<std::endl;
    }
}
double compute_inner_product_fast(const double *x, data_info d){
    //Try to eliminate some inefficiencies in the code, no more explicit transposing
    //Optimize matrix products to reduce the number of multiplications all in a singls
    //function.
    //
    //data_info d = *((struct data_info *)data);
    double max,sum,squarediff, dp;
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
            dp = d.diag2[i]/max - d._ip_mat[i][j];
            d.stored_dp[r] = dp;
            squarediff += pow(dp, 2);
            //squarediff += pow((d.diag2[i] - d._ip_mat[i][j]), 2);
        }
        else{
            for (int k = 0 ; k < d.ndim ; k++ ){
                sum += d.Z[k] * (d.M1[i][k]*d.M1[j][k]);
            
                // VALID ONLY FOR d.ndim == 3!!!!!!!!!
                t = (k*2)%d.ndim;
                s = (k*2+1)%d.ndim;

                sum += d.Z[k+3]*(d.M1[i][t]*d.M1[j][s] + d.M1[i][s]*d.M1[j][t]);
            }
            dp = sum/d.diag[i]/d.diag[j] - d._ip_mat[i][j];
            d.stored_dp[r] = dp;
            squarediff += pow(dp, 2);
            //squarediff += pow((sum - d._ip_mat[i][j]), 2);
            sum=0;
        }
    }
    return squarediff;
}


void jacobian3D_sums(double* grad, const double* x, data_info d) {
    //data_info d = *((struct data_info *)data);
    int i, j, m, n, cocycle_start,z_ind;
    double sum, max; 
    double dp1,dp2,dp3,dp4;
    max=d.diag2[d.diag_ind];
    //Z should already be created from the objective function
    //create_metric_tensor(d.ndim, d.farray, d.Z);
    //the metric tensor first
    //Something wrong with this as the x parameters are scaled to form the
    // Z matrix...
    for (int sz=0; sz<6; sz++){
        if(sz<3){
            i=sz;
            j=sz;
            
        }

        else if(sz==3){ 
            i=0;
            j=1;
            
        }
        else if(sz==4){
            i=0;
            j=2;
            
        }
        else if (sz==5){
            i=1;
            j=2;
            
        }
        //std::cout<<sz<<" "<<"Index: "<<i<<" "<<j<<std::endl;
        grad[sz]=0.0;
        for (int r = 0 ; r < d.nz_size ; r++ ){
            m = d._zi[r];
            n = d._zj[r];
            dp1=0.0;
            dp2=0.0;
            dp3=0.0;
            dp4=0.0;
            for (int size=0; size<d.B_shape; size++){
                dp1 += d._cycle_cocycle_I[m][size]*d.rep[size][i];
                dp2 += d._cycle_cocycle_I[n][size]*d.rep[size][j];
                dp3 += d._cycle_cocycle_I[n][size]*d.rep[size][i];
                dp4 += d._cycle_cocycle_I[m][size]*d.rep[size][j];
            }
            sum = dp1*dp2 + dp3*dp4; 
            if(m==n){
                grad[sz] += 2.0*(d.stored_dp[r])*sum/max;
                //grad[sz] += sum;
            }
            else{
                grad[sz] += 2.0*(d.stored_dp[r])*sum/d.diag[m]/d.diag[n];
                //grad[sz] += sum; 
            } 
        }
        //std::cout<<grad[sz]<<std::endl;
    }


    //then the other entries
    cocycle_start=d.cycle_size;
    for (int sz=6; sz<d.x_size; sz++){
        i = (cocycle_start) + (sz-6)/3;
        j = sz%3;
        grad[sz]=0.0;
        for (int r = 0 ; r < d.nz_size ; r++ ){
            m = d._zi[r];
            n = d._zj[r];
            sum=0.0;
            for (int dim=0; dim<3; dim++){
                dp1=0.0;
                dp2=0.0;
                if(dim==j)z_ind=dim;
                else z_ind=j+dim+2;
                for (int size=0; size<d.B_shape; size++){
                    dp1 += d._cycle_cocycle_I[n][size]*d.rep[size][dim];
                    dp2 += d._cycle_cocycle_I[m][size]*d.rep[size][dim];
                }
                sum += (d._cycle_cocycle_I[m][i]*dp1 + d._cycle_cocycle_I[n][i]*dp2)*d.Z[z_ind];
            }
            if(m==n){
                grad[sz] += 2.0*(d.stored_dp[r])*sum/max;
                //grad[sz] += sum/max;
            }
            else {
                grad[sz] += 2.0*(d.stored_dp[r])*sum/d.diag[m]/d.diag[n]; 
                //grad[sz] += sum/d.diag[m]/d.diag[n];
            }
        }
        //std::cout<<grad[sz]<<std::endl;
    }
}


double objectivefunc2D(unsigned n, const double *x, double *grad, void *dd)
{
    double ans;
    data_info d = *((struct data_info *)dd); 
    create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, x, d.rep);
    create_metric_tensor2D(d.ndim, x, d.Z);
    ans = compute_inner_product_fast2D(x, d);
    if (grad) {
        //Jacobian calc not working!!
        //jacobian3D_sums(grad, x, d);
        //forward_difference_grad(grad, x, ans, dd, 1e-5);
        central_difference_grad2D(grad, x, dd, 1e-5);
    }
    //std::cout<<ans<<std::endl;
    return ans; 
}

void central_difference_grad2D(double* grad, const double* x, void* data, double xinc){
    data_info d = *((struct data_info *)data);
    double forward, back;
    for (int i=0; i<d.x_size; i++){
        memcpy((void*)d.farray, (void*)x, d.x_size*sizeof(double));
        memcpy((void*)d.barray, (void*)x, d.x_size*sizeof(double));
        d.farray[i] += xinc;
        d.barray[i] -= xinc;
        //forward grad
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, d.farray, d.rep);
        create_metric_tensor2D(d.ndim, d.farray, d.Z);
        forward = compute_inner_product_fast2D(d.farray, d);

        //backward grad
        create_full_rep(d.cycle_size, d.ndim, d._cycle_rep, d.start, d.x_size/d.ndim, d.barray, d.rep);
        create_metric_tensor2D(d.ndim, d.barray, d.Z);
        back = compute_inner_product_fast2D(d.barray,d);
        grad[i] = (forward - back)/(2.*xinc);
        //std::cout<<grad[i]<<std::endl;
    }
}
double compute_inner_product_fast2D(const double *x, data_info d){
    //Try to eliminate some inefficiencies in the code, no more explicit transposing
    //Optimize matrix products to reduce the number of multiplications all in a singls
    //function.
    //
    //data_info d = *((struct data_info *)data);
    double max,sum,squarediff, dp;
    int i, j;
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

        sum+= d.Z[0] * d.M1[r][0]*d.M1[r][0];
        sum+= d.Z[1] * d.M1[r][1]*d.M1[r][1];
        sum+= d.Z[2] * 2.*d.M1[r][0]*d.M1[r][1];
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
            dp = d.diag2[i]/max - d._ip_mat[i][j];
            d.stored_dp[r] = dp;
            squarediff += pow(dp, 2);
            //squarediff += pow((d.diag2[i] - d._ip_mat[i][j]), 2);
        }
        else{

            sum += d.Z[0]*d.M1[i][0]*d.M1[j][0];
            sum += d.Z[1]*d.M1[i][1]*d.M1[j][1];
            sum += d.Z[2]*(d.M1[i][1]*d.M1[j][0] + d.M1[i][0]*d.M1[j][1]);
            dp = sum/d.diag[i]/d.diag[j] - d._ip_mat[i][j];
            d.stored_dp[r] = dp;
            squarediff += pow(dp, 2);
            //squarediff += pow((sum - d._ip_mat[i][j]), 2);
            sum=0;
        }
    }
    return squarediff;
}
