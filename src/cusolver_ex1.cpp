// nvcc src/cusolver_ex1.cpp -o ex -L/usr/local/cuda/lib64 -lcusparse -lcusolver -lcudart && ./ex

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverDn.h"
#include "cusolverSp.h"

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

int printDmat(int m, int n, int l, double *mat, bool device)
{
    double *matLoc;
    if (device)
    {
        matLoc = (double *)malloc(sizeof(double) * l * n);
        CHECK_CUDA(cudaMemcpy(matLoc, mat, sizeof(double) * l * n, cudaMemcpyDeviceToHost));
    }
    else
    {
        matLoc = mat;
    }

    for (int i = 0; i < l; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%6.2lf ", matLoc[l * i + j]);
        }
        printf("\n");
    }
    printf("\n");

    if (device)
    {
        free(matLoc);
    }

    return 0;
}

int min(int a, int b)
{
    return a < b ? a : b;
}

int main()
{
    int i;
    int nrows = 2;
    int ncols = 2;
    int ld = 2;
    int info;

    int *d_ipiv = NULL;

    int bufferSize;

    double h_A[] = {3.0, 0, 2, 0};
    double *d_A = NULL;
    double *d_buffer = NULL;

    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnCreate(&cusolverDnHandle);

    CHECK_CUDA(cudaMalloc((void **)&d_A, sizeof(double) * ld * ncols))
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(double) * ld * ncols, cudaMemcpyHostToDevice))

    cusolverDnDgetrf_bufferSize(cusolverDnHandle,
                                nrows, ncols,
                                d_A, ld,
                                &bufferSize);

    printf("getrf buffer size: %lu\n", bufferSize);

    CHECK_CUDA(cudaMalloc((void **)&d_buffer, bufferSize))
    CHECK_CUDA(cudaMalloc((void **)&d_ipiv, sizeof(int) * min(nrows, ncols)))

    cusolverDnDgetrf(cusolverDnHandle,
                     nrows, ncols,
                     d_A, ld,
                     (double *)d_buffer, d_ipiv,
                     &info);

    printf("getrf info: %d\n", info);
    printf("A after getrf\n");
    printDmat(nrows, ncols, ld, d_A, true);

    // set I matrix
    double *h_I = (double *)calloc(min(nrows, ncols) * min(nrows, ncols), sizeof(double));
    for (i = 0; i < min(nrows, ncols); ++i)
    {
        h_I[min(nrows, ncols) * i + i] = 1.0;
    }
    CHECK_CUDA(cudaMemcpy(d_buffer, h_I, sizeof(double) * min(nrows, ncols) * min(nrows, ncols), cudaMemcpyHostToDevice))
    free(h_I);

    cusolverDnDgetrs(cusolverDnHandle, CUBLAS_OP_N,
                     nrows, ncols,
                     d_A, ld,
                     d_ipiv,
                     (double *)d_buffer, nrows,
                     &info);

    printf("getrs info: %d\n", info);
    printf("A after getrs\n");
    printDmat(nrows, ncols, ld, d_A, true);

    cudaFree(d_A);
    cudaFree(d_buffer);
    cudaFree(d_ipiv);

    return 0;
}