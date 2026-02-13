
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

/** Step-length ratios: t_prim[j] = (dx[j]<0) ? -x[j]/dx[j] : DBL_MAX, same for dual. */
__global__ void alpha_max_ratios_kernel(const double *x, const double *dx,
                                        const double *s, const double *ds,
                                        double *t_prim, double *t_dual, int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N)
        return;
    double dxj = dx[j];
    double dsj = ds[j];
    t_prim[j] = (dxj < 0.0) ? (-x[j] / dxj) : DBL_MAX;
    t_dual[j] = (dsj < 0.0) ? (-s[j] / dsj) : DBL_MAX;
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

/** Compute alphaMaxPrim = min(-x/delta_x over delta_x<0), alphaMaxDual = min(-s/delta_s over delta_s<0) on device.
 *  Caller must provide device buffers: d_tmp_prim, d_tmp_dual (length N), d_blockmin_prim, d_blockmin_dual (length nBlocks). */
void alpha_max_dev(const double *d_x, const double *d_deltaX, const double *d_s, const double *d_deltaS,
                   int N,
                   double *d_tmp_prim, double *d_tmp_dual,
                   double *d_blockmin_prim, double *d_blockmin_dual,
                   double *alphaMaxPrim, double *alphaMaxDual)
{
    const int blockDim = 256;
    const int gridDim = (N + blockDim - 1) / blockDim;
    const size_t shmem = blockDim * sizeof(double);

    alpha_max_ratios_kernel<<<gridDim, blockDim>>>(d_x, d_deltaX, d_s, d_deltaS, d_tmp_prim, d_tmp_dual, N);
    computeMinimum_kernel<<<gridDim, blockDim, shmem>>>(d_tmp_prim, d_blockmin_prim, (unsigned int)N);
    computeMinimum_kernel<<<gridDim, blockDim, shmem>>>(d_tmp_dual, d_blockmin_dual, (unsigned int)N);
    cudaDeviceSynchronize();

    /* Final min on host over block results */
    double h_prim[1024], h_dual[1024];
    if (gridDim > 1024)
    {
        cudaDeviceSynchronize();
        return; /* fallback: leave *alphaMaxPrim/Dual unchanged */
    }
    cudaMemcpy(h_prim, d_blockmin_prim, gridDim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dual, d_blockmin_dual, gridDim * sizeof(double), cudaMemcpyDeviceToHost);
    *alphaMaxPrim = h_prim[0];
    *alphaMaxDual = h_dual[0];
    for (int i = 1; i < gridDim; i++)
    {
        if (h_prim[i] < *alphaMaxPrim)
            *alphaMaxPrim = h_prim[i];
        if (h_dual[i] < *alphaMaxDual)
            *alphaMaxDual = h_dual[i];
    }
}
