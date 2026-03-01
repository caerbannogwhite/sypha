#include "sypha_solver.h"
#include "sypha_solver_sparse.h"
#include "sypha_node_sparse.h"
#include "sypha_cuda_helper.h"
#include <cstdint>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"

// CUDA 12+ cuSPARSE: legacy enums removed; use generic API enums
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
#ifndef CUSPARSE_CSRMV_ALG2
#define CUSPARSE_CSRMV_ALG2 CUSPARSE_SPMV_CSR_ALG2
#endif
#ifndef CUSPARSE_CSR2CSC_ALG2
#define CUSPARSE_CSR2CSC_ALG2 CUSPARSE_CSR2CSC_ALG1
#endif
#ifndef CUSPARSE_CSRMM_ALG1
#define CUSPARSE_CSRMM_ALG1 CUSPARSE_SPMM_ALG_DEFAULT
#endif
#endif

SyphaStatus solver_sparse_mehrotra_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    size_t currBufferSize = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = nullptr;
    double *d_bufferS = nullptr;
    double *d_buffer = nullptr;

    double *d_resC = nullptr, *d_resB = nullptr, *d_resXS = nullptr;
    double *d_x = nullptr, *d_y = nullptr, *d_s = nullptr;
    double *d_delX = nullptr, *d_delY = nullptr, *d_delS = nullptr;

    cusparseSpMatDescr_t spMatTransDescr;
    cusparseDnVecDescr_t vecX, vecY, vecResC, vecResB;

    ///////////////////             GET TRANSPOSED MATRIX

    int *d_csrMatTransOffs = nullptr, *d_csrMatTransInds = nullptr;
    double *d_csrMatTransVals = nullptr;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_csrMatTransOffs), sizeof(int) * (node.ncols + 1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_csrMatTransInds), sizeof(int) * node.nnz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_csrMatTransVals), sizeof(double) * node.nnz));

    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                                  node.dCsrMatVals, node.dCsrMatOffs, node.dCsrMatInds,
                                                  d_csrMatTransVals, d_csrMatTransOffs, d_csrMatTransInds,
                                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                                  &bufferSize));
    // buffer size for other needs
    currBufferSize = static_cast<size_t>(sizeof(double) * node.ncols * 2);
    currBufferSize = currBufferSize > bufferSize ? currBufferSize : bufferSize;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));

    checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                       node.dCsrMatVals, node.dCsrMatOffs, node.dCsrMatInds,
                                       d_csrMatTransVals, d_csrMatTransOffs, d_csrMatTransInds,
                                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                       CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                       d_buffer));

    checkCudaErrors(cusparseCreateCsr(&spMatTransDescr, node.ncols, node.nrows, node.nnz,
                                      d_csrMatTransOffs, d_csrMatTransInds, d_csrMatTransVals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.hX.resize(node.ncols);
    node.hY.resize(node.nrows);
    node.hS.resize(node.ncols);

    node.timeStartSolStart = node.env->timer();
    solver_sparse_mehrotra_init_gsl(node);
    node.timeStartSolEnd = node.env->timer();

    ///////////////////             INITIALISE RHS

    node.timePreSolStart = node.env->timer();

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_s), sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_x, node.hX.data(), sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_y, node.hY.data(), sizeof(double) * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_s, node.hS.data(), sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));

    // put OBJ and S on device rhs
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_resC), sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_resB), sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_resXS), sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_resC, node.dObjDns, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_resB, node.dRhsDns, sizeof(double) * node.nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

    // Residuals
    // resB, resC equation 14.7, page 395(414)Numerical Optimization
    // resC = -mat' * y + (obj - s)
    // resB = -mat  * x + rhs

    checkCudaErrors(cusparseCreateDnVec(&vecX, static_cast<int64_t>(node.ncols), d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, static_cast<int64_t>(node.nrows), d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResC, static_cast<int64_t>(node.ncols), d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, static_cast<int64_t>(node.nrows), d_resB, CUDA_R_64F));

    alpha = -1.0;
    checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                &alpha, d_s, 1, d_resC, 1));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, spMatTransDescr, vecY,
                                            &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    // buffer size for other needs
    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, spMatTransDescr, vecY,
                                 &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, node.matDescr, vecX,
                                            &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, node.matDescr, vecX,
                                 &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 (size_t *)d_buffer));

    ///////////////////             CALCULATE MU
    // duality measure, defined at page 395(414) Numerical Optimization
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
    mu /= node.ncols;

    node.timePreSolEnd = node.env->timer();

    ///////////////////             MAIN LOOP

    if (node.env->getLogger())
        node.env->getLogger()->log(LOG_TRACE, "Mehrotra procedure started");
    node.timeSolverStart = node.env->timer();

    iterations = 0;

    while ((iterations < node.env->getMehrotraMaxIter()) && (mu > node.env->getMehrotraMuTol()))
    {

        ++iterations;
    }

    node.timeSolverEnd = node.env->timer();

    ///////////////////             RELEASE RESOURCES

    cusparseDestroySpMat(spMatTransDescr);

    checkCudaErrors(cudaFree(d_csrMatTransOffs));
    checkCudaErrors(cudaFree(d_csrMatTransInds));
    checkCudaErrors(cudaFree(d_csrMatTransVals));

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_s));

    checkCudaErrors(cudaFree(d_resC));
    checkCudaErrors(cudaFree(d_resB));
    checkCudaErrors(cudaFree(d_resXS));

    return CODE_SUCCESSFUL;
}

#if !(defined(CUDART_VERSION) && CUDART_VERSION >= 12000)
SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int64_t AAT_nrows = node.nrows, AAT_ncols = node.nrows, AAT_nnz = 0;
    double alpha = 1.0;
    double beta = 0.0;

    int *AAT_inds = nullptr, *AAT_offs = nullptr;
    double *AAT_vals = nullptr;

    void *d_buffer1 = nullptr, *d_buffer2 = nullptr;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    cusparseSpMatDescr_t AAT_descr;
    cusparseMatDescr_t AAT_descrGen, matTransDescrGen;
    cusparseSpGEMMDescr_t spgemmDescr;

    checkCudaErrors(cusparseCreateMatDescr(&AAT_descrGen));
    checkCudaErrors(cusparseSetMatType(AAT_descrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(AAT_descrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateMatDescr(&matTransDescrGen));
    checkCudaErrors(cusparseSetMatType(matTransDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matTransDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    ///////////////////             COMPUTE STARTING COORDINATES X AND S

    // AAT matrix for geMM
    checkCudaErrors(cusparseCreateCsr(&AAT_descr, AAT_nrows, AAT_ncols, AAT_nnz,
                                      nullptr, nullptr, nullptr,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // SpGEMM Computation
    checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

    // ask bufferSize1 bytes for external memory
    checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, node.matTransDescr,
                                                  &beta, AAT_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, nullptr));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer1), bufferSize1));

    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, node.matTransDescr,
                                                  &beta, AAT_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, d_buffer1));

    // ask bufferSize2 bytes for external memory
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, node.matTransDescr,
                                           &beta, AAT_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, nullptr));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer2), bufferSize2));

    // compute the intermediate product of A * B
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, node.matTransDescr,
                                           &beta, AAT_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

    // get matrix C non-zero entries C_num_nnz1
    // cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);
    cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);

    // allocate matrix AAT
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&AAT_offs), sizeof(int) * (AAT_nrows + 1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&AAT_inds), sizeof(int) * AAT_nnz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&AAT_vals), sizeof(double) * AAT_nnz));

    // update AAT with the new pointers
    checkCudaErrors(cusparseCsrSetPointers(AAT_descr, AAT_offs, AAT_inds, AAT_vals));

    // copy the final products to the matrix AAT
    checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, node.matDescr, node.matTransDescr,
                                        &beta, AAT_descr,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

    checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));

    ///////////////////             COMPUTE s = - mat' * y + obj
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.dS, node.dObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseCsrmvEx_bufferSize(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               node.ncols, node.nrows, node.nnz,
                                               &alpha, CUDA_R_64F,
                                               matTransDescrGen,
                                               node.dCsrMatTransVals, CUDA_R_64F,
                                               node.dCsrMatTransOffs,
                                               node.dCsrMatTransInds,
                                               node.dY, CUDA_R_64F,
                                               &beta, CUDA_R_64F,
                                               node.dS, CUDA_R_64F, CUDA_R_64F,
                                               &bufferSize1));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer1), bufferSize1));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseCsrmvEx(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    node.ncols, node.nrows, node.nnz,
                                    &alpha, CUDA_R_64F,
                                    matTransDescrGen,
                                    node.dCsrMatTransVals, CUDA_R_64F,
                                    node.dCsrMatTransOffs,
                                    node.dCsrMatTransInds,
                                    node.dY, CUDA_R_64F,
                                    &beta, CUDA_R_64F,
                                    node.dS, CUDA_R_64F, CUDA_R_64F,
                                    d_buffer1));

    checkCudaErrors(cusparseDestroyMatDescr(AAT_descrGen));
    checkCudaErrors(cusparseDestroyMatDescr(matTransDescrGen));
    checkCudaErrors(cusparseDestroySpMat(AAT_descr));

    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));

    checkCudaErrors(cudaFree(AAT_inds));
    checkCudaErrors(cudaFree(AAT_offs));
    checkCudaErrors(cudaFree(AAT_vals));

    return CODE_SUCCESSFUL;
}
#endif // !(CUDA 12+)

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
// CUDA 12+: legacy csrmvEx / CUSPARSE_ALG_MERGE_PATH are unavailable.
// Reuse the GSL-based initialisation instead of the legacy cuSPARSE path.
SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node)
{
    return solver_sparse_mehrotra_init_gsl(node);
}
#endif

SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;
    int info = 0;
    int i = 0;
    int I_matBytes = node.nrows * node.nrows * sizeof(double);

    double alpha = 1.0;
    double beta = 0.0;

    int *d_ipiv = nullptr;
    double *d_AAT = nullptr;
    double *d_matDn = nullptr;
    double *h_I = nullptr;

    void *d_buffer = nullptr;
    size_t currBufferSize = 0;
    size_t bufferSize = 0;

    cusolverDnParams_t cusolverDnParams;
    cusparseDnVecDescr_t vecX, vecY, vecS;
    cusparseDnMatDescr_t AAT_descr, matDnDescr;
    cusparseMatDescr_t matDescrGen;

    if (node.env->getLogger())
        node.env->getLogger()->log(LOG_DEBUG, "Computing Mehrotra starting point");
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    checkCudaErrors(cusparseCreateMatDescr(&matDescrGen));
    checkCudaErrors(cusparseSetMatType(matDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateDnVec(&vecX, static_cast<int64_t>(node.ncols),
                                        node.dX, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecY, static_cast<int64_t>(node.nrows),
                                        node.dY, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecS, static_cast<int64_t>(node.ncols),
                                        node.dS, CUDA_R_64F));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_AAT), sizeof(double) * node.nrows * node.nrows));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matDn), sizeof(double) * node.nrows * node.ncols));

    checkCudaErrors(cusparseCreateDnMat(&AAT_descr, static_cast<int64_t>(node.nrows), static_cast<int64_t>(node.nrows),
                                        static_cast<int64_t>(node.nrows), d_AAT, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    checkCudaErrors(cusparseCreateDnMat(&matDnDescr, static_cast<int64_t>(node.nrows), static_cast<int64_t>(node.ncols),
                                        static_cast<int64_t>(node.nrows), d_matDn, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    ///////////////////             STORE MATRIX IN DENSE FORMAT
    if (node.env->getLogger())
        node.env->getLogger()->log(LOG_TRACE, "Init: storing matrix (dense)");
    size_t szSparseToDense = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    checkCudaErrors(cusparseSparseToDense_bufferSize(node.cusparseHandle, node.matDescr, matDnDescr,
                                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &szSparseToDense));
#endif
    checkCudaErrors(cusparseSpMM_bufferSize(node.cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, matDnDescr,
                                            &beta, AAT_descr,
                                            CUDA_R_64F,
                                            CUSPARSE_CSRMM_ALG1,
                                            &bufferSize));

    // allocate memory for computation
    currBufferSize = bufferSize > I_matBytes ? bufferSize : I_matBytes;
    if (szSparseToDense > currBufferSize)
        currBufferSize = szSparseToDense;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    checkCudaErrors(cusparseSparseToDense(node.cusparseHandle, node.matDescr, matDnDescr,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buffer));
#else
    checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, node.nrows, node.ncols,
                                       matDescrGen,
                                       node.dCsrMatVals, node.dCsrMatOffs, node.dCsrMatInds,
                                       d_matDn, node.nrows));
#endif

    ///////////////////             COMPUTE AAT INVERSE MATRIX

    // GEMM Computation: MATRIX * MATRIX'

    checkCudaErrors(cusparseSpMM(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, matDnDescr,
                                 &beta, AAT_descr,
                                 CUDA_R_64F,
                                 CUSPARSE_CSRMM_ALG1,
                                 d_buffer));

    ///////////////////             MATRIX INVERSION

    // See https://stackoverflow.com/questions/50892906/what-is-the-most-efficient-way-to-compute-the-inverse-of-a-general-matrix-using
    checkCudaErrors(cusolverDnDgetrf_bufferSize(node.cusolverDnHandle,
                                                node.nrows, node.nrows,
                                                d_AAT, node.nrows,
                                                (int *)&bufferSize));

    // allocate memory for computation
    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));
    }
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ipiv), sizeof(int) * node.nrows));

    // set I matrix
    h_I = (double *)calloc(node.nrows * node.nrows, sizeof(double));
    for (i = 0; i < node.nrows; ++i)
    {
        h_I[node.nrows * i + i] = 1.0;
    }
    // checkCudaErrors(cudaMemcpyAsync(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    // checkCudaErrors(cudaMemcpy(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice));
    free(h_I);

    checkCudaErrors(cusolverDnDgetrs(node.cusolverDnHandle, CUBLAS_OP_N,
                                     node.nrows, node.nrows,
                                     d_AAT, node.nrows,
                                     d_ipiv,
                                     (double *)d_buffer, node.nrows,
                                     &info));

    ///////////////////             COMPUTE s = - mat' * y + obj
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.dS, node.dObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, vecY,
                                            &beta, vecS,
                                            CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, vecY,
                                 &beta, vecS,
                                 CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    ///////////////////             FREE RESOURCES
    checkCudaErrors(cusolverDnDestroyParams(cusolverDnParams));

    checkCudaErrors(cusparseDestroyMatDescr(matDescrGen));
    checkCudaErrors(cusparseDestroyDnMat(AAT_descr));

    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecS));

    checkCudaErrors(cudaFree(d_ipiv));
    checkCudaErrors(cudaFree(d_buffer));

    checkCudaErrors(cudaFree(d_AAT));
    checkCudaErrors(cudaFree(d_matDn));

    return CODE_SUCCESSFUL;
}

SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node)
{
    int i, j;
    int signum = 0;
    double deltaX, deltaS, prod, sumX, sumS;

    gsl_vector *x = nullptr;
    gsl_vector *y = nullptr;
    gsl_vector *s = nullptr;
    gsl_matrix *inv = nullptr;
    gsl_matrix *mat = nullptr;
    gsl_matrix *tmp = nullptr;
    gsl_permutation *perm = nullptr;

    x = gsl_vector_alloc(static_cast<size_t>(node.ncols));
    y = gsl_vector_alloc(static_cast<size_t>(node.nrows));
    s = gsl_vector_alloc(static_cast<size_t>(node.ncols));
    inv = gsl_matrix_calloc(static_cast<size_t>(node.nrows), static_cast<size_t>(node.nrows));
    mat = gsl_matrix_calloc(static_cast<size_t>(node.nrows), static_cast<size_t>(node.ncols));
    tmp = gsl_matrix_calloc(static_cast<size_t>(node.nrows), static_cast<size_t>(node.ncols));
    perm = gsl_permutation_alloc(static_cast<size_t>(node.nrows));

    // csr to dense
    for (i = 0; i < node.nrows; ++i)
    {
        for (j = node.hCsrMatOffs.data()[i]; j < node.hCsrMatOffs.data()[i + 1]; ++j)
        {
            mat->data[node.ncols * i + node.hCsrMatInds.data()[j]] = node.hCsrMatVals.data()[j];
        }
    }

    ///////////////////             MATRIX MULT
    mat->size1 = node.nrows;
    mat->size2 = node.ncols;
    mat->tda = node.ncols;
    tmp->size1 = node.nrows;
    tmp->size2 = node.nrows;
    tmp->tda = node.ncols;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mat, mat, 0.0, tmp);

    ///////////////////             MATRIX INVERSION

    inv->size1 = node.nrows;
    inv->size2 = node.nrows;
    inv->tda = node.nrows;
    gsl_linalg_LU_decomp(tmp, perm, &signum);
    gsl_linalg_LU_invert(tmp, perm, inv);

    ///////////////////             COMPUTE x = mat' * AAT_inv * rhs

    tmp->size1 = node.ncols;
    tmp->size2 = node.nrows;
    tmp->tda = node.nrows;
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mat, inv, 0.0, tmp);

    // put RHS in Y
    memcpy(y->data, node.hRhsDns.data(), sizeof(double) * node.nrows);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, y, 0.0, x);

    ///////////////////             COMPUTE y = AAT_inv * mat * obj

    tmp->size1 = node.nrows;
    tmp->size2 = node.ncols;
    tmp->tda = node.ncols;

    // put OBJ in S
    memcpy(s->data, node.hObjDns.data(), sizeof(double) * node.ncols);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, mat, 0.0, tmp);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, s, 0.0, y);

    ///////////////////             COMPUTE s = - mat' * y + obj
    gsl_blas_dgemv(CblasTrans, -1.0, mat, y, 1.0, s);

    deltaX = gsl_max(-1.5 * gsl_vector_min(x), 0.0);
    deltaS = gsl_max(-1.5 * gsl_vector_min(s), 0.0);

    gsl_vector_add_constant(x, deltaX);
    gsl_vector_add_constant(s, deltaS);

    gsl_blas_ddot(x, s, &prod);
    prod *= 0.5;

    sumX = 0.0;
    sumS = 0.0;
    for (j = 0; j < node.ncols; ++j)
    {
        sumX += x->data[j];
        sumS += s->data[j];
    }
    deltaX = prod / sumS;
    deltaS = prod / sumX;

    gsl_vector_add_constant(x, deltaX);
    gsl_vector_add_constant(s, deltaS);

    memcpy(node.hX.data(), x->data, sizeof(double) * node.ncols);
    memcpy(node.hY.data(), y->data, sizeof(double) * node.nrows);
    memcpy(node.hS.data(), s->data, sizeof(double) * node.ncols);

    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(s);
    gsl_matrix_free(inv);
    gsl_matrix_free(mat);
    gsl_matrix_free(tmp);
    gsl_permutation_free(perm);

    return CODE_SUCCESSFUL;
}
