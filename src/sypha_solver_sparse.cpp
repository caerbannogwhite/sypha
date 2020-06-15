
#include "sypha_solver_sparse.h"

SyphaStatus solver_sparse_merhrotra(SyphaNodeSparse &node)
{
    int iterations = 0;
    double mu;

    solver_sparse_merhrotra_init(node);

    while ((iterations < node.env->MERHROTRA_MAX_ITER) && (mu > node.env->MERHROTRA_MU_TOL))
    {

        ++iterations;
    }

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_merhrotra_init(SyphaNodeSparse &node)
{
    double alpha = 1.0;
    double beta = 0.0;

    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    cusparseSpMatDescr_t matC;
    cusparseSpGEMMDescr_t spgemmDesc;

    ////////////////////        COMPUTE STARTING COORDINATES X AND S

    // fake C matrix for geMM
    cusparseCreateCsr(&matC, node.numRows, node.numRows, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // SpGEMM Computation
    cusparseSpGEMM_createDescr(&spgemmDesc);

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_TRANSPOSE,
                                  &alpha, node.matDescr, node.matDescr,
                                  &beta, matC,
                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);

    cudaMalloc((void **)&dBuffer1, bufferSize1);

    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_TRANSPOSE,
                                  &alpha, node.matDescr, node.matDescr,
                                  &beta, matC,
                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(node.cusparseHandle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_TRANSPOSE,
                           &alpha, node.matDescr, node.matDescr,
                           &beta, matC,
                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);

    cudaMalloc((void **)&dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(node.cusparseHandle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_TRANSPOSE,
                           &alpha, node.matDescr, node.matDescr,
                           &beta, matC,
                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);

    std::cout << "rows: " << node.numRows << ", cols: " << node.numCols << std::endl;
    std::cout << "buff size 1: " << bufferSize1 << ", buff size 2: " << bufferSize2 << std::endl; 

    return CODE_SUCCESFULL;
}