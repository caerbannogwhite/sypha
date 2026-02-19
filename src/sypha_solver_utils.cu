
#include "sypha_solver_utils.h"

__global__ void elem_min_mult_kernel(double *d_A, double *d_B, double *d_C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_C[idx] = -d_A[idx] * d_B[idx];
}

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, int N, cudaStream_t stream)
{
    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;
    elem_min_mult_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N);
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
                       double *d_out, int N, cudaStream_t stream)
{
    const int blockDim = 256;
    const int gridDim = (N + blockDim - 1) / blockDim;
    corrector_rhs_kernel<<<gridDim, blockDim, 0, stream>>>(d_deltaX, d_deltaS, sigma, mu, d_out, N);
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

/** Final reduction: reduce nBlocks block-level minimums to a single scalar.
 *  Launched with gridDim=1, blockDim=next_power_of_2(nBlocks). */
__global__ void finalMinimum_kernel(const double *blockResults, double *result, int nBlocks)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;

    double val = DBL_MAX;
    if ((int)tid < nBlocks)
    {
        val = blockResults[tid];
    }
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
        result[0] = sdata[0];
}

void alpha_max_dev(const double *d_x, const double *d_deltaX, const double *d_s, const double *d_deltaS,
                   int N,
                   double *d_tmp_prim, double *d_tmp_dual,
                   double *d_blockmin_prim, double *d_blockmin_dual,
                   double *d_alphaResult,
                   double *alphaMaxPrim, double *alphaMaxDual,
                   cudaStream_t stream)
{
    const int blockSize = 256;
    const int nBlocks = (N + blockSize - 1) / blockSize;
    const size_t shmem = blockSize * sizeof(double);

    if (nBlocks > 1024)
    {
        return; /* safety: leave outputs unchanged */
    }

    // Step 1: compute per-element ratios
    alpha_max_ratios_kernel<<<nBlocks, blockSize, 0, stream>>>(d_x, d_deltaX, d_s, d_deltaS, d_tmp_prim, d_tmp_dual, N);

    // Step 2: per-block reduction
    computeMinimum_kernel<<<nBlocks, blockSize, shmem, stream>>>(d_tmp_prim, d_blockmin_prim, (unsigned int)N);
    computeMinimum_kernel<<<nBlocks, blockSize, shmem, stream>>>(d_tmp_dual, d_blockmin_dual, (unsigned int)N);

    // Step 3: final reduction on GPU (1 block)
    int finalBlockSize = 1;
    while (finalBlockSize < nBlocks) finalBlockSize <<= 1;
    if (finalBlockSize < 1) finalBlockSize = 1;
    size_t finalShmem = finalBlockSize * sizeof(double);
    finalMinimum_kernel<<<1, finalBlockSize, finalShmem, stream>>>(d_blockmin_prim, &d_alphaResult[0], nBlocks);
    finalMinimum_kernel<<<1, finalBlockSize, finalShmem, stream>>>(d_blockmin_dual, &d_alphaResult[1], nBlocks);

    // Step 4: copy just 2 doubles to host
    double h_result[2];
    cudaMemcpyAsync(h_result, d_alphaResult, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    *alphaMaxPrim = h_result[0];
    *alphaMaxDual = h_result[1];
}
