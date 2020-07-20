
#include "sypha_solver_dense.h"

SyphaStatus solver_dense_mehrotra(SyphaNodeDense &node)
{
    int i = 0, j = 0, k = 0, nnz = 0, iterations = 0;
    size_t bufferSize = 0;
    double mu = 0.0;
    void *d_buffer = NULL;

    ///////////////////             GET TRANSPOSED MATRIX
    

    ///////////////////             TEST
    
    ///////////////////             GET STARTING POINT

    solver_dense_mehrotra_init(node);

    while ((iterations < node.env->MEHROTRA_MAX_ITER) && (mu > node.env->MEHROTRA_MU_TOL))
    {

        ++iterations;
    }


    ///////////////////             RELEASE RESOURCES


    return CODE_SUCCESFULL;
}

SyphaStatus solver_dense_mehrotra_init(SyphaNodeDense &node)
{

    ///////////////////             COMPUTE STARTING COORDINATES X AND S

    
    return CODE_SUCCESFULL;
}