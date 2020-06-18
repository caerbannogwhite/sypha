/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include csrms2.cpp 
 *   g++ -o csrm2 csrsm2.o -L/usr/local/cuda/lib64 -lcusparse -lcudart
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
/* compute | b - A*x|_inf */
void residaul_eval(
    int n,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    const float *x,
    float *r_nrminf_ptr)
{
    const int base = (cusparseGetMatIndexBase(descrA) !=
                      CUSPARSE_INDEX_BASE_ONE)
                         ? 0
                         : 1;
    const int lower = (CUSPARSE_FILL_MODE_LOWER ==
                       cusparseGetMatFillMode(descrA))
                          ? 1
                          : 0;
    const int unit = (CUSPARSE_DIAG_TYPE_UNIT ==
                      cusparseGetMatDiagType(descrA))
                         ? 1
                         : 0;
    float r_nrminf = 0;
    for (int row = 0; row < n; row++)
    {
        const int start = csrRowPtr[row] - base;
        const int end = csrRowPtr[row + 1] - base;
        float dot = 0;
        for (int colidx = start; colidx < end; colidx++)
        {
            const int col = csrColInd[colidx] - base;
            float Aij = csrVal[colidx];
            float xj = x[col];
            if ((row == col) && unit)
            {
                Aij = 1.0;
            }
            int valid = (row >= col) && lower ||
                        (row <= col) && !lower;
            if (valid)
            {
                dot += Aij * xj;
            }
        }
        float ri = b[row] - dot;
        r_nrminf = (r_nrminf > fabs(ri)) ? r_nrminf : fabs(ri);
    }
    *r_nrminf_ptr = r_nrminf;
}
int main(int argc, char *argv[])
{
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL;
    csrsm2Info_t info = NULL;

    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    const int nrhs = 2;
    const int n = 4;
    const int nnzA = 9;
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const float h_one = 1.0;
    /* 
 *      |    1     0     2     -3  |
 *      |    0     4     0     0   |
 *  A = |    5     0     6     7   |
 *      |    0     8     0     9   |
 *
 *  Regard A as a lower triangle matrix L with non-unit diagonal.
 *             | 1  5 |              | 1          5      |
 *  Given  B = | 2  6 |, X = L \ B = | 0.5       1.5     |
 *             | 3  7 |              | -0.3333   -3      |
 *             | 4  8 |              |  0        -0.4444 |
 */
    const int csrRowPtrA[n + 1] = {1, 4, 5, 8, 10};
    const int csrColIndA[nnzA] = {1, 3, 4, 2, 1, 3, 4, 2, 4};
    const float csrValA[nnzA] = {1, 2, -3, 4, 5, 6, 7, 8, 9};
    const float B[n * nrhs] = {1, 2, 3, 4, 5, 6, 7, 8};
    float X[n * nrhs];
    int *d_csrRowPtrA = NULL;
    int *d_csrColIndA = NULL;
    float *d_csrValA = NULL;
    float *d_B = NULL;
    size_t lworkInBytes = 0;
    char *d_work = NULL;
    const int algo = 0; /* non-block version */
    printf("example of csrsm2 \n");
    /* step 1: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);
    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    status = cusparseCreateCsrsm2Info(&info);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    /* step 2: configuration of matrix A */
    status = cusparseCreateMatDescr(&descrA);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    /* A is base-1*/
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    /* A is lower triangle */
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
    /* A has non unit diagonal */
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cudaStat1 = cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (n + 1));
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc((void **)&d_csrValA, sizeof(float) * nnzA);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMalloc((void **)&d_B, sizeof(float) * n * nrhs);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (n + 1),
                           cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA,
                           cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(d_csrValA, csrValA, sizeof(float) * nnzA,
                           cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(d_B, B, sizeof(float) * n * nrhs,
                           cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    /* step 3: query workspace */
    status = cusparseScsrsm2_bufferSizeExt(
        handle,
        algo,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
        n,
        nrhs,
        nnzA,
        &h_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_B,
        n, /* ldb */
        info,
        policy,
        &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    printf("lworkInBytes  = %lld \n", (long long)lworkInBytes);
    if (NULL != d_work)
    {
        cudaFree(d_work);
    }
    cudaStat1 = cudaMalloc((void **)&d_work, lworkInBytes);
    assert(cudaSuccess == cudaStat1);
    /* step 4: analysis */
    status = cusparseScsrsm2_analysis(
        handle,
        algo,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
        n,
        nrhs,
        nnzA,
        &h_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_B,
        n, /* ldb */
        info,
        policy,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    /* step 5: solve L * X = B */
    status = cusparseScsrsm2_solve(
        handle,
        algo,
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transA */
        CUSPARSE_OPERATION_NON_TRANSPOSE, /* transB */
        n,
        nrhs,
        nnzA,
        &h_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_B,
        n, /* ldb */
        info,
        policy,
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    /* step 6:measure residual B - A*X */
    cudaStat1 = cudaMemcpy(X, d_B, sizeof(float) * n * nrhs,
                           cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    cudaDeviceSynchronize();
    printf("==== x1 = inv(A)*b1 \n");
    for (int j = 0; j < n; j++)
    {
        printf("x1[%d] = %f\n", j, X[j]);
    }
    float r1_nrminf;
    residaul_eval(
        n,
        descrA,
        csrValA,
        csrRowPtrA,
        csrColIndA,
        B,
        X,
        &r1_nrminf);
    printf("|b1 - A*x1| = %E\n", r1_nrminf);
    printf("==== x2 = inv(A)*b2 \n");
    for (int j = 0; j < n; j++)
    {
        printf("x2[%d] = %f\n", j, X[n + j]);
    }
    float r2_nrminf;
    residaul_eval(
        n,
        descrA,
        csrValA,
        csrRowPtrA,
        csrColIndA,
        B + n,
        X + n,
        &r2_nrminf);
    printf("|b2 - A*x2| = %E\n", r2_nrminf);
    /* free resources */
    if (d_csrRowPtrA)
        cudaFree(d_csrRowPtrA);
    if (d_csrColIndA)
        cudaFree(d_csrColIndA);
    if (d_csrValA)
        cudaFree(d_csrValA);
    if (d_B)
        cudaFree(d_B);
    if (handle)
        cusparseDestroy(handle);
    if (stream)
        cudaStreamDestroy(stream);
    if (descrA)
        cusparseDestroyMatDescr(descrA);
    if (info)
        cusparseDestroyCsrsm2Info(info);
    cudaDeviceReset();
    return 0;
}