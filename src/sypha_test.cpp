#include "sypha_test.h"

int sypha_test_001()
{
    cudaStream_t cudaStream;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusolverDnHandle_t cusolverDnHandle;
    cusolverSpHandle_t cusolverSpHandle;

    // initialize a cuda stream for this node
    checkCudaErrors(cudaStreamCreate(&cudaStream));

    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cusparseCreate(&cusparseHandle));
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverSpCreate(&cusolverSpHandle));

    // bind stream to cusparse and cusolver
    checkCudaErrors(cublasSetStream(cublasHandle, cudaStream));
    checkCudaErrors(cusparseSetStream(cusparseHandle, cudaStream));
    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, cudaStream));
    checkCudaErrors(cusolverSpSetStream(cusolverSpHandle, cudaStream));

    ///////////////////             TEST PARAMS
    int reorder = 0;
    int singularity = 0;

    double SING_TOL = 1.E-16;

    // int A_nrows = 3;
    // int A_ncols = 3;
    // int A_nnz = 4;

    // int h_csrAOffs[] = {0, 2, 3, 4};
    // int h_csrAInds[] = {0, 2, 1, 0};
    // double h_csrAVals[] = {1.0, 1.0, 1.0, 1.0};
    // double h_b[] = {1.0, 1.0, 1.0};
    // double *h_x = (double *)malloc(sizeof(double) * A_nrows);

    int A_nrows = 12;
    int A_ncols = 12;
    int A_nnz = 25;

    int h_csrAOffs[] = {   0,    2,    4,    6,    8,   10,   13,   15,   17,   19,   21,   23,   25 };
    int h_csrAInds[] = {   5,    7,    6,    8,    5,    9,    5,   10,    6,   11,    0,    2,    3,    1,    4,    0,    7,    1,    8,    2,    9,    3,   10,    4,   11};
    double h_csrAVals[] = { 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, -1.000000, 1.000000, -1.000000, 1.000000, 1.000000, 1.000000, -1.000000, 1.000000, -1.000000, 0.275510, 1.458333, 0.775510, 1.625000, 1.275510, 1.458333, 1.275510, 0.791667, 0.775510, 0.625000};
    double h_b[] = {-0.275510, -0.275510, -0.275510, -0.275510, -0.275510, -1.125000, 0.000000, -0.401786, -1.260204, -1.860119, -1.009779, -0.484694};
    double *h_x = (double *)malloc(sizeof(double) * A_nrows);

    int *d_csrAInds = NULL;
    int *d_csrAOffs = NULL;
    double *d_csrAVals = NULL;

    double *d_b = NULL;
    double *d_x = NULL;

    cusparseMatDescr_t A_descr;

    checkCudaErrors(cusparseCreateMatDescr(&A_descr));
    checkCudaErrors(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cudaMalloc((void **)&d_csrAInds, sizeof(int) * A_nnz));
    checkCudaErrors(cudaMalloc((void **)&d_csrAOffs, sizeof(int) * (A_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrAVals, sizeof(double) * A_nnz));

    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double) * A_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * A_nrows));

    checkCudaErrors(cudaMemcpy(d_csrAInds, h_csrAInds, sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAOffs, h_csrAOffs, sizeof(int) * (A_nrows + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAVals, h_csrAVals, sizeof(double) * A_nnz, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double) * A_nrows, cudaMemcpyHostToDevice));

    printf("SYPHA TEST - Solving system with QR [GPU] cusolverSpDcsrlsvqr (rows: %d, cols: %d, nnz: %d)\n", A_nrows, A_ncols, A_nnz);
    checkCudaErrors(cusolverSpDcsrlsvqr(cusolverSpHandle,
                                        A_nrows, A_nnz, A_descr,
                                        d_csrAVals, d_csrAOffs, d_csrAInds,
                                        d_b,
                                        SING_TOL, reorder,
                                        d_x, &singularity));

    printf("SYPHA TEST - Solution (singularity: %d)\n", singularity);
    utils_printDvec(A_nrows, d_x, true);

    printf("SYPHA TEST - Solving system with CHOL [GPU] cusolverSpDcsrlsvchol (rows: %d, cols: %d, nnz: %d)\n", A_nrows, A_ncols, A_nnz);
    checkCudaErrors(cusolverSpDcsrlsvchol(cusolverSpHandle,
                                          A_nrows, A_nnz, A_descr,
                                          d_csrAVals, d_csrAOffs, d_csrAInds,
                                          d_b,
                                          SING_TOL, reorder,
                                          d_x, &singularity));

    printf("SYPHA TEST - Solution (singularity: %d)\n", singularity);
    utils_printDvec(A_nrows, d_x, true);

    printf("SYPHA TEST - Solving system with QR [CPU] cusolverSpDcsrlsvqrHost (rows: %d, cols: %d, nnz: %d)\n", A_nrows, A_ncols, A_nnz);
    checkCudaErrors(cusolverSpDcsrlsvqrHost(cusolverSpHandle,
                                        A_nrows, A_nnz, A_descr,
                                        h_csrAVals, h_csrAOffs, h_csrAInds,
                                        h_b,
                                        SING_TOL, reorder,
                                        h_x, &singularity));

    printf("SYPHA TEST - Solution (singularity: %d)\n", singularity);
    utils_printDvec(A_nrows, h_x, false);

    printf("SYPHA TEST - Solving system with CHOL [CPU] cusolverSpDcsrlsvcholHost (rows: %d, cols: %d, nnz: %d)\n", A_nrows, A_ncols, A_nnz);
    checkCudaErrors(cusolverSpDcsrlsvcholHost(cusolverSpHandle,
                                          A_nrows, A_nnz, A_descr,
                                          h_csrAVals, h_csrAOffs, h_csrAInds,
                                          h_b,
                                          SING_TOL, reorder,
                                          h_x, &singularity));

    printf("SYPHA TEST - Solution (singularity: %d)\n", singularity);
    utils_printDvec(A_nrows, h_x, false);

    free(h_x);

    checkCudaErrors(cudaFree(d_csrAInds));
    checkCudaErrors(cudaFree(d_csrAOffs));
    checkCudaErrors(cudaFree(d_csrAVals));

    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_x));

    if (A_descr)
        checkCudaErrors(cusparseDestroyMatDescr(A_descr));

    if (cublasHandle)
        checkCudaErrors(cublasDestroy(cublasHandle));
    if (cusparseHandle)
        checkCudaErrors(cusparseDestroy(cusparseHandle));
    if (cusolverDnHandle)
        checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
    if (cusolverSpHandle)
        checkCudaErrors(cusolverSpDestroy(cusolverSpHandle));

    if (cudaStream)
        checkCudaErrors(cudaStreamDestroy(cudaStream));
}