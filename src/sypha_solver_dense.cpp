/**
 * Dense Mehrotra interior point solver (stub).
 * The active solver is the sparse path (SyphaNodeSparse / solver_sparse_mehrotra).
 */
#include "sypha_solver_dense.h"

SyphaStatus solver_dense_mehrotra(SyphaNodeDense &node)
{
    int iterations = 0;
    double mu = 0.0;

    solver_dense_mehrotra_init(node);

    while ((iterations < node.env->MEHROTRA_MAX_ITER) && (mu > node.env->MEHROTRA_MU_TOL))
    {
        ++iterations;
    }

    return CODE_SUCCESFULL;
}

SyphaStatus solver_dense_mehrotra_init(SyphaNodeDense &node)
{
    (void)node;
    return CODE_SUCCESFULL;
}