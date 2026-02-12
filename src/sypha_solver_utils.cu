
#include "sypha_solver_utils.h"

// __global__ void elementMultiplyVector_kernel(double *x, double *y, double *z, unsigned int N)
// {
//     unsigned int idx;
//     for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
//         z[idx] = x[idx] * y[idx];
// }
//
// void elementMultiplyVector(double *x, double *y, double *z, unsigned int N)
// {
//     dim3 blockDim (128);
//     dim3 gridDim (min((N - 1) / blockDim.x + 1, 32 * 1024));
//     elementMultiplyVector_kernel <<<gridDim, blockDim>>>(x, y, z, N);
// }

__global__ void elem_min_mult_kernel(double *d_A, double *d_B, double *d_C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_C[idx] = -d_A[idx] * d_B[idx];
}

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, int N)
{
    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;
    elem_min_mult_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}

void elem_min_mult_host(double *d_A, double *d_B, double *d_C, const int N)
{
    double *buffA;
    double *buffB;
    double *buffC;

    buffA = (double *)malloc(sizeof(double) * N);
    buffB = (double *)malloc(sizeof(double) * N);
    buffC = (double *)malloc(sizeof(double) * N);

    cudaMemcpy(buffA, d_A, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffB, d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);

    for (int j = 0; j < N; ++j)
    {
        buffC[j] = -buffA[j] * buffB[j];
    }

    cudaMemcpy(d_C, buffC, sizeof(double) * N, cudaMemcpyHostToDevice);

    free(buffA);
    free(buffB);
    free(buffC);
}

void elem_min_mult_hybr(double *d_A, double *d_B, double *d_C, int N)
{
    int off = 0;
    if (N >= 128)
    {
        int bSize = N / 128;
        off = bSize * 32; /* kernel covers bSize blocks * 32 threads */
        elem_min_mult_kernel<<<bSize, 32>>>(d_A, d_B, d_C, off);
        N -= off;
    }
    if (N > 0)
        elem_min_mult_host(&d_A[off], &d_B[off], &d_C[off], N);
    cudaDeviceSynchronize();
}

__global__ void corrector_rhs_kernel(const double *d_deltaX, const double *d_deltaS,
                                     double sigma, double mu, double *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_out[idx] = -d_deltaX[idx] * d_deltaS[idx] + sigma * mu;
}

void corrector_rhs_dev(double *d_deltaX, double *d_deltaS, double sigma, double mu,
                       double *d_out, int N)
{
    const int blockDim = 256;
    const int gridDim = (N + blockDim - 1) / blockDim;
    corrector_rhs_kernel<<<gridDim, blockDim>>>(d_deltaX, d_deltaS, sigma, mu, d_out, N);
    cudaDeviceSynchronize();
}

__global__ void computeMinimum_kernel(double *x, double *result, const unsigned int N)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    double val = DBL_MAX;
    if (i < N)
    {
        val = x[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // Find min
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write local minimum
    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}
