
#include "sypha_solver_sparse.h"

SyphaStatus solver_sparse_merhrotra(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    double mu = 0.0;
    void *d_buffer = NULL;
    char message[1024];

    cusparseMatDescr_t A_descr;

    ///////////////////             GET TRANSPOSED MATRIX
    // checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransOffs, sizeof(int) * (node.ncols + 1)));
    // checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransInds, sizeof(int) * node.nnz));
    // checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransVals, sizeof(double) * node.nnz));

    // checkCudaErrors(cudaDeviceSynchronize());

    // checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
    //                                               node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
    //                                               node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
    //                                               CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
    //                                               CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
    //                                               &bufferSize));

    // checkCudaErrors(cudaMalloc((void **)&d_buffer, bufferSize));

    // checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
    //                                    node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
    //                                    node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
    //                                    CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
    //                                    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
    //                                    d_buffer));

    // checkCudaErrors(cusparseCreateCsr(&node.matTransDescr, node.ncols, node.nrows, node.nnz,
    //                                   //node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
    //                                   node.d_csrMatTransOffs, node.d_csrMatTransInds, node.d_csrMatTransVals,
    //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.h_x = (double *)malloc(sizeof(double) * node.ncols);
    node.h_y = (double *)malloc(sizeof(double) * node.nrows);
    node.h_s = (double *)malloc(sizeof(double) * node.ncols);

    solver_sparse_merhrotra_init_gsl(node);


    ///////////////////             SET BIG MATRIX ON HOST
    //
    // On each step we solve this linear system twice:
    //
    //      O | A' | I    x    -rc
    //      --|----|---   -    ---
    //      A | O  | O  * y  = -rb
    //      --|----|---   -    ---
    //      S | O  | X    s    -rxs
    //
    // Where A is the model matrix (standard form), I is the n*n identity
    // matrix, S is the n*n s diagonal matrix, X is the n*n diagonal matrix.
    // Total number of non-zero elements is A.nnz * 2 + n * 3

    int A_nrows = node.ncols * 2 + node.nrows;
    int A_ncols = A_nrows;
    int A_nnz = node.nnz * 2 + node.ncols * 3;
    
    int *h_csrAInds = NULL;
    int *h_csrAOffs = NULL;
    double *h_csrAVals = NULL;

    int *d_csrAInds = NULL;
    int *d_csrAOffs = NULL;
    double *d_csrAVals = NULL;

    double *d_rhs = NULL;
    double *d_sol = NULL;

    h_csrAInds = (int *)calloc(sizeof(int), A_nnz);
    h_csrAOffs = (int *)calloc(sizeof(int), (A_nrows + 1));
    h_csrAVals = (double *)calloc(sizeof(double), A_nnz);

    sprintf(message, "Initialising matrix: %d rows, %d columns, %d non zeros", A_nrows, A_ncols, A_nnz);
    node.env->logger(message, "INFO", 17);

    // Instantiate the first group of n rows: O | A' | I
    bool found = false;
    int off = 0, rowCnt = 0;

    h_csrAOffs[0] = 0;
    for (j = 0; j < node.ncols; ++j)
    {
        rowCnt = 0;
        for (i = 0; i < node.nrows; ++i)
        {
            found = false;
            for (k = node.h_csrMatOffs->data()[i]; k < node.h_csrMatOffs->data()[i+1]; ++k)
            {
                if (node.h_csrMatInds->data()[k] == j)
                {
                    found = true;
                    break;
                }
            }

            if (found)
            {
                h_csrAInds[off] = node.ncols + i;
                h_csrAVals[off] = node.h_csrMatVals->data()[k];
                ++rowCnt;
                ++off;
            }
        }

        // append the I matrix element for the current row
        h_csrAInds[off] = node.ncols + node.nrows + j;
        h_csrAVals[off] = 1.0;
        ++rowCnt;
        ++off;

        h_csrAOffs[j + 1] = h_csrAOffs[j] + rowCnt;
    }

    // Instantiate the second group of m rows: A | O | O
    for (i = 0; i < node.nrows; ++i)
    {
        h_csrAOffs[node.ncols + i + 1] = h_csrAOffs[node.ncols + i] + (node.h_csrMatOffs->data()[i + 1] - node.h_csrMatOffs->data()[i]);
    }
    memcpy(&h_csrAInds[off], node.h_csrMatInds->data(), sizeof(int) * node.nnz);
    memcpy(&h_csrAVals[off], node.h_csrMatVals->data(), sizeof(double) * node.nnz);
    off += node.nnz;

    // Instantiate the third group of n rows: S | O | X
    for (j = 0; j < node.ncols; ++j)
    {
        // s
        h_csrAInds[off] = j;
        h_csrAVals[off] = node.h_s[j];
        ++off;

        // x
        h_csrAInds[off] = node.ncols + node.nrows + j;
        h_csrAVals[off] = node.h_x[j];
        ++off;

        h_csrAOffs[node.ncols + node.nrows + j + 1] = h_csrAOffs[node.ncols + node.nrows + j] + 2;
    }

    checkCudaErrors(cudaMalloc((void **)&d_csrAInds, sizeof(int) * A_nnz));
    checkCudaErrors(cudaMalloc((void **)&d_csrAOffs, sizeof(int) * (A_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrAVals, sizeof(double) * A_nnz));

    checkCudaErrors(cudaMemcpy(d_csrAInds, h_csrAInds, sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAOffs, h_csrAOffs, sizeof(int) * (A_nrows + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAVals, h_csrAVals, sizeof(double) * A_nnz, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cusparseCreateMatDescr(&A_descr));
    checkCudaErrors(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));
    
    ///////////////////             TEST
    /* 
    double *h_ADn = NULL, *d_ADn = NULL;
    h_ADn = (double *)calloc(sizeof(double), A_nrows * A_ncols);
    checkCudaErrors(cudaMalloc((void **)&d_ADn, sizeof(double) * A_nrows * A_ncols));

    checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, A_nrows, A_ncols,
                                       A_descr, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
                                       d_csrAVals, d_csrAOffs, d_csrAInds,
                                       d_ADn, A_nrows));

    checkCudaErrors(cudaMemcpy(h_ADn, d_ADn, sizeof(double) * A_nrows * A_ncols, cudaMemcpyDeviceToHost));
    //for (i = 0; i < A_nrows; ++i)
    //{
    //    for (k = h_csrAOffs[i]; k < h_csrAOffs[i+1]; ++k)
    //    {
    //        h_ADn[i*A_nrows+h_csrAInds[k]] = h_csrAVals[k];
    //    }
    //}
    utils_printDmat(A_nrows, A_ncols, A_nrows, h_ADn, false);

    free(h_ADn);
    checkCudaErrors(cudaFree(d_ADn));
    */
    ///////////////////             END TEST

    free(h_csrAInds);
    free(h_csrAOffs);
    free(h_csrAVals);

    checkCudaErrors(cudaMalloc((void **)&d_rhs, sizeof(double) * A_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_sol, sizeof(double) * A_nrows));

    node.env->logger("Starting Merhrotra proceduce", "INFO", 17);
    while ((iterations < node.env->MERHROTRA_MAX_ITER) && (mu > node.env->MERHROTRA_MU_TOL))
    {
        checkCudaErrors(cusolverSpDcsrlsvchol(node.cusolverSpHandle,
                                              A_nrows, A_nnz, A_descr,
                                              d_csrAVals, d_csrAOffs, d_csrAInds,
                                              d_rhs,
                                              node.env->MERHROTRA_CHOL_TOL, reorder,
                                              d_sol, &singularity));

        checkCudaErrors(cusolverSpDcsrlsvchol(node.cusolverSpHandle,
                                              A_nrows, A_nnz, A_descr,
                                              d_csrAVals, d_csrAOffs, d_csrAInds,
                                              d_rhs,
                                              node.env->MERHROTRA_CHOL_TOL, reorder,
                                              d_sol, &singularity));

        ++iterations;
    }

    ///////////////////             RELEASE RESOURCES
    checkCudaErrors(cusparseDestroyMatDescr(A_descr));

    checkCudaErrors(cudaFree(d_csrAInds));
    checkCudaErrors(cudaFree(d_csrAOffs));
    checkCudaErrors(cudaFree(d_csrAVals));

    checkCudaErrors(cudaFree(d_rhs));
    checkCudaErrors(cudaFree(d_sol));

    checkCudaErrors(cusparseDestroySpMat(node.matTransDescr));
    node.matTransDescr = NULL;

    // checkCudaErrors(cudaFree(node.d_csrMatTransInds));
    // checkCudaErrors(cudaFree(node.d_csrMatTransOffs));
    // checkCudaErrors(cudaFree(node.d_csrMatTransVals));

    // node.d_csrMatTransInds = NULL;
    // node.d_csrMatTransOffs = NULL;
    // node.d_csrMatTransVals = NULL;

    // checkCudaErrors(cudaFree(d_buffer));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_merhrotra_init_1(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int64_t AAT_nrows = node.nrows, AAT_ncols = node.nrows, AAT_nnz = 0;
    double alpha = 1.0;
    double beta = 0.0;

    int *AAT_inds = NULL, *AAT_offs = NULL;
    double *AAT_vals = NULL;

    void *d_buffer1 = NULL, *d_buffer2 = NULL;
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
                                      NULL, NULL, NULL,
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
                                                  spgemmDescr, &bufferSize1, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, bufferSize1));

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
                                           spgemmDescr, &bufferSize2, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_buffer2, bufferSize2));

    // compute the intermediate product of A * B
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, node.matTransDescr,
                                           &beta, AAT_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

    // get matrix C non-zero entries C_num_nnz1
    //cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);
    cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);

    // allocate matrix AAT
    checkCudaErrors(cudaMalloc((void **)&AAT_offs, sizeof(int) * (AAT_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&AAT_inds, sizeof(int) * AAT_nnz));
    checkCudaErrors(cudaMalloc((void **)&AAT_vals, sizeof(double) * AAT_nnz));

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

    int64_t r, c, n;
    std::cout << "buff 1: " << bufferSize1 << ", buff 2: " << bufferSize2 << std::endl;

    cusparseSpMatGetSize(node.matDescr, &r, &c, &n);
    std::cout << "\nMat" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;
    cusparseSpMatGetSize(node.matTransDescr, &r, &c, &n);
    std::cout << "\nTrans" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;
    cusparseSpMatGetSize(AAT_descr, &r, &c, &n);
    std::cout << "\nAAT" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;

    /*void *d_b, *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_b, AAT_nrows*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_x, AAT_nrows*sizeof(double)));

    checkCudaErrors(cusolverSpDcsrlsvchol(
        node.cusolverSpHandle, AAT_nrows, AAT_nnz,
        AAT_descrGen, AAT_vals, AAT_offs, AAT_inds,
        (double*)d_b, node.env->MERHROTRA_CHOL_TOL, reorder, (double*)d_x, &singularity));

    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_x));*/

    cusparseSpMatGetSize(AAT_descr, &r, &c, &n);
    std::cout << "\nAAT" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;

    ///////////////////             COMPUTE s = - mat' * y + obj
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.d_s, node.d_ObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseCsrmvEx_bufferSize(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               node.ncols, node.nrows, node.nnz,
                                               &alpha, CUDA_R_64F,
                                               matTransDescrGen,
                                               node.d_csrMatTransVals, CUDA_R_64F,
                                               node.d_csrMatTransOffs,
                                               node.d_csrMatTransInds,
                                               node.d_y, CUDA_R_64F,
                                               &beta, CUDA_R_64F,
                                               node.d_s, CUDA_R_64F, CUDA_R_64F,
                                               &bufferSize1));

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, bufferSize1));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseCsrmvEx(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    node.ncols, node.nrows, node.nnz,
                                    &alpha, CUDA_R_64F,
                                    matTransDescrGen,
                                    node.d_csrMatTransVals, CUDA_R_64F,
                                    node.d_csrMatTransOffs,
                                    node.d_csrMatTransInds,
                                    node.d_y, CUDA_R_64F,
                                    &beta, CUDA_R_64F,
                                    node.d_s, CUDA_R_64F, CUDA_R_64F,
                                    d_buffer1));

    checkCudaErrors(cusparseDestroyMatDescr(AAT_descrGen));
    checkCudaErrors(cusparseDestroyMatDescr(matTransDescrGen));
    checkCudaErrors(cusparseDestroySpMat(AAT_descr));

    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));

    checkCudaErrors(cudaFree(AAT_inds));
    checkCudaErrors(cudaFree(AAT_offs));
    checkCudaErrors(cudaFree(AAT_vals));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_merhrotra_init_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;
    int info = 0;
    int i = 0;
    int I_matBytes = node.nrows * node.nrows * sizeof(double);

    double alpha = 1.0;
    double beta = 0.0;

    int *d_ipiv = NULL;
    double *d_AAT = NULL;
    double *d_matDn = NULL;
    double *h_I = NULL;

    void *d_buffer = NULL;
    size_t currBufferSize = 0;
    size_t bufferSize = 0;
    char message[1024];

    cusolverDnParams_t cusolverDnParams;
    cusparseDnVecDescr_t vecX, vecY, vecS;
    cusparseDnMatDescr_t AAT_descr, matDnDescr;
    cusparseMatDescr_t matDescrGen;

    node.env->logger("Merhrotra starting point computation", "INFO", 13);
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    checkCudaErrors(cusparseCreateMatDescr(&matDescrGen));
    checkCudaErrors(cusparseSetMatType(matDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols,
                                        node.d_x, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows,
                                        node.d_y, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecS, (int64_t)node.ncols,
                                        node.d_s, CUDA_R_64F));

    checkCudaErrors(cudaMalloc((void **)&d_AAT, sizeof(double) * node.nrows * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_matDn, sizeof(double) * node.nrows * node.ncols));

    checkCudaErrors(cusparseCreateDnMat(&AAT_descr, (int64_t)node.nrows, (int64_t)node.nrows,
                                        (int64_t)node.nrows, d_AAT, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    checkCudaErrors(cusparseCreateDnMat(&matDnDescr, (int64_t)node.nrows, (int64_t)node.ncols,
                                        (int64_t)node.nrows, d_matDn, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    ///////////////////             STORE MATRIX IN DENSE FORMAT
    node.env->logger("solver_sparse_merhrotra_init - storing matrix in dense format", "INFO", 20);
    checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, node.nrows, node.ncols,
                                       matDescrGen, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
                                       node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                       d_matDn, node.nrows));

    ///////////////////             COMPUTE AAT INVERSE MATRIX

    // GEMM Computation: MATRIX * MATRIX'
    node.env->logger("solver_sparse_merhrotra_init - computing mat * mat'", "INFO", 20);
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
    checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));

    checkCudaErrors(cusparseSpMM(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, matDnDescr,
                                 &beta, AAT_descr,
                                 CUDA_R_64F,
                                 CUSPARSE_CSRMM_ALG1,
                                 d_buffer));

    ///////////////////             MATRIX INVERSION

    node.env->logger("solver_sparse_merhrotra_init - computing matrix inversion", "INFO", 20);
    // See https://stackoverflow.com/questions/50892906/what-is-the-most-efficient-way-to-compute-the-inverse-of-a-general-matrix-using
    checkCudaErrors(cusolverDnDgetrf_bufferSize(node.cusolverDnHandle,
                                                node.nrows, node.nrows,
                                                d_AAT, node.nrows,
                                                (int *)&bufferSize));

    // allocate memory for computation
    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }
    checkCudaErrors(cudaMalloc((void **)&d_ipiv, sizeof(int) * node.nrows));

    /*checkCudaErrors(cusolverDnDgetrf(node.cusolverDnHandle,
                                     node.nrows, node.nrows,
                                     d_AAT, node.nrows,
                                     (double *)d_buffer, d_ipiv,
                                     &info));*/

    printf("AAT after getrf\n");
    utils_printDmat(node.nrows, node.nrows, node.nrows, d_AAT, true);

    sprintf(message, "solver_sparse_merhrotra_init - cusolverDnGetrf returned %d", info);
    node.env->logger(message, "INFO", 20);

    // set I matrix
    h_I = (double *)calloc(node.nrows * node.nrows, sizeof(double));
    for (i = 0; i < node.nrows; ++i)
    {
        h_I[node.nrows * i + i] = 1.0;
    }
    //checkCudaErrors(cudaMemcpyAsync(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    //checkCudaErrors(cudaMemcpy(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice));
    free(h_I);

    checkCudaErrors(cusolverDnDgetrs(node.cusolverDnHandle, CUBLAS_OP_N,
                                     node.nrows, node.nrows,
                                     d_AAT, node.nrows,
                                     d_ipiv,
                                     (double *)d_buffer, node.nrows,
                                     &info));

    printf("AAT after getrs\n");
    utils_printDmat(node.nrows, node.nrows, node.nrows, d_AAT, true);

    sprintf(message, "solver_sparse_merhrotra_init - cusolverDnGetrs returned %d", info);
    node.env->logger(message, "INFO", 20);

    /*void *d_b, *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_b, AAT_nrows*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_x, AAT_nrows*sizeof(double)));

    checkCudaErrors(cusolverSpDcsrlsvchol(
        node.cusolverSpHandle, AAT_nrows, AAT_nnz,
        matDescrGen, AAT_vals, AAT_offs, AAT_inds,
        (double*)d_b, node.env->MERHROTRA_CHOL_TOL, reorder, (double*)d_x, &singularity));

    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_x));*/

    ///////////////////             COMPUTE s = - mat' * y + obj
    node.env->logger("solver_sparse_merhrotra_init - computing s = - mat' * y + obj", "INFO", 20);
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.d_s, node.d_ObjDns, sizeof(double) * node.ncols,
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
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
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

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_merhrotra_init_gsl(SyphaNodeSparse &node)
{
    int i, j;
    int signum = 0;
    double deltaX, deltaS, prod, sumX, sumS;
    char message[1024];

    gsl_vector *x = NULL;
    gsl_vector *y = NULL;
    gsl_vector *s = NULL;
    gsl_matrix *inv = NULL;
    gsl_matrix *mat = NULL;
    gsl_matrix *tmp = NULL;
    gsl_permutation *perm = NULL;

    x = gsl_vector_alloc((size_t)node.ncols);
    y = gsl_vector_alloc((size_t)node.nrows);
    s = gsl_vector_alloc((size_t)node.ncols);
    inv = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.nrows);
    mat = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.ncols);
    tmp = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.ncols);
    perm = gsl_permutation_alloc((size_t)node.nrows);

    // csr to dense
    for (i = 0; i < node.nrows; ++i)
    {
        for (j = node.h_csrMatOffs->data()[i]; j < node.h_csrMatOffs->data()[i + 1]; ++j)
        {
            mat->data[node.ncols * i + node.h_csrMatInds->data()[j]] = node.h_csrMatVals->data()[j];
        }
    }
    //printf("MAT:\n");
    //utils_printDmat(node.nrows, node.ncols, node.ncols, mat->data, false);

    ///////////////////             MATRIX MULT
    node.env->logger("solver_sparse_merhrotra_init - computing A * A'", "INFO", 20);
    mat->size1 = node.nrows;
    mat->size2 = node.ncols;
    mat->tda = node.ncols;
    tmp->size1 = node.nrows;
    tmp->size2 = node.nrows;
    tmp->tda = node.ncols;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mat, mat, 0.0, tmp);

    //printf("AAT:\n");
    //utils_printDmat(node.nrows, node.nrows, node.ncols, tmp->data, false);

    ///////////////////             MATRIX INVERSION
    node.env->logger("solver_sparse_merhrotra_init - computing inv(AAT)", "INFO", 20);
    
    inv->size1 = node.nrows;
    inv->size2 = node.nrows;
    inv->tda = node.nrows;
    gsl_linalg_LU_decomp(tmp, perm, &signum);
    gsl_linalg_LU_invert(tmp, perm, inv);

    //printf("INV:\n");
    //utils_printDmat(node.nrows, node.nrows, node.nrows, inv->data, false);

    ///////////////////             COMPUTE x = mat' * AAT_inv * rhs
    node.env->logger("solver_sparse_merhrotra_init - computing x <-- A' * inv(AAT) * rhs", "INFO", 20);

    tmp->size1 = node.ncols;
    tmp->size2 = node.nrows;
    tmp->tda = node.nrows;
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mat, inv, 0.0, tmp);

    // put RHS in Y
    memcpy(y->data, node.h_RhsDns, sizeof(double) * node.nrows);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, y, 0.0, x);

    //printf("TMP:\n");
    //utils_printDmat(node.ncols, node.nrows, node.nrows, tmp->data, false);

    ///////////////////             COMPUTE y = AAT_inv * mat * obj
    node.env->logger("solver_sparse_merhrotra_init - computing y <-- inv(AAT) * A * obj", "INFO", 20);

    tmp->size1 = node.nrows;
    tmp->size2 = node.ncols;
    tmp->tda = node.ncols;
    
    // put OBJ in S
    memcpy(s->data, node.h_ObjDns, sizeof(double) * node.ncols);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, mat, 0.0, tmp);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, s, 0.0, y);

    //printf("TMP:\n");
    //utils_printDmat(node.ncols, node.nrows, node.nrows, tmp->data, false);

    ///////////////////             COMPUTE s = - mat' * y + obj
    node.env->logger("solver_sparse_merhrotra_init - computing s <-- obj - A' * y", "INFO", 20);
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

    //printf("X:\n");
    //utils_printDvec(node.ncols, x->data, false);
    //printf("Y:\n");
    //utils_printDvec(node.nrows, y->data, false);
    //printf("S:\n");
    //utils_printDvec(node.ncols, s->data, false);

    memcpy(node.h_x, x->data, sizeof(double) * node.ncols);
    memcpy(node.h_y, y->data, sizeof(double) * node.nrows);
    memcpy(node.h_s, s->data, sizeof(double) * node.ncols);

    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(s);
    gsl_matrix_free(inv);
    gsl_matrix_free(mat);
    gsl_matrix_free(tmp);
    gsl_permutation_free(perm);

    return CODE_SUCCESFULL;
}
