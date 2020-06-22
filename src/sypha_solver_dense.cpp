
#include "sypha_solver_dense.h"

SyphaStatus solver_dense_merhrotra(SyphaNodeDense &node)
{
    int i = 0, j = 0, k = 0, nnz = 0, iterations = 0;
    size_t bufferSize = 0;
    double mu = 0.0;
    void *d_buffer = NULL;

    ///////////////////             GET TRANSPOSED MATRIX
    

    ///////////////////             TEST
    
    ///////////////////             GET STARTING POINT

    solver_dense_merhrotra_init(node);

    while ((iterations < node.env->MERHROTRA_MAX_ITER) && (mu > node.env->MERHROTRA_MU_TOL))
    {

        ++iterations;
    }


    ///////////////////             RELEASE RESOURCES


    return CODE_SUCCESFULL;
}

SyphaStatus solver_dense_merhrotra_init(SyphaNodeDense &node)
{

    ///////////////////             COMPUTE STARTING COORDINATES X AND S

    
    return CODE_SUCCESFULL;
}