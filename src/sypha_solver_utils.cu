
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

__global__ void range_kernel(int *d_A, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		d_A[idx] = idx;
}

__global__ void range_kernel(double *d_A, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		d_A[idx] = idx;
}

__global__ void elem_mult_kernel(double *d_A, double *d_B, double *d_C, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		d_C[idx] = d_A[idx] * d_B[idx];
}

__global__ void elem_min_mult_kernel(double *d_A, double *d_B, double *d_C, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		d_C[idx] = -d_A[idx] * d_B[idx];
}

__global__ void elem_inv_kernel(double *d_A, double *d_invA, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		d_invA[idx] = 1.0 / d_A[idx];
}

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, const int N)
{	
	int bSize = (N >> 5) + 1;
	elem_min_mult_kernel<<<bSize, 32>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();
}

void elem_min_mult_host(double *d_A, double *d_B, double *d_C, const int N)
{
	double *buffA;
	double *buffB;
	double *buffC;
	
	buffA = (double*)malloc(sizeof(double) * N);
	buffB = (double*)malloc(sizeof(double) * N);
	buffC = (double*)malloc(sizeof(double) * N);
	
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
		off = bSize * 128;
		N -= off;
		elem_min_mult_kernel<<<bSize, 32>>>(d_A, d_B, d_C, N);
	}
	
	elem_min_mult_host(&d_A[off], &d_B[off], &d_C[off], N);
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
            sdata[tid] = fmin(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }
    
    // Write local minimum
    if (tid == 0) result[blockIdx.x] = sdata[0];
}


void find_alpha_max(double *alphaMaxPrim, double *alphaMaxDual,
					double *d_x, double *d_deltaX, double *d_s, double *d_deltaS, const int N)
{
	double alpha, beta;

	*alphaMaxPrim = DBL_MAX;
	*alphaMaxDual = DBL_MAX;
	
	for (int j = 0; j < N; ++j)
	{
		checkCudaErrors(cudaMemcpy(&alpha, &d_x[j], sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&beta, &d_deltaX[j], sizeof(double), cudaMemcpyDeviceToHost));
		if (beta < 0.0)
		{
			alpha = -(alpha / beta);
			*alphaMaxPrim = *alphaMaxPrim < alpha ? *alphaMaxPrim : alpha;
		}

		checkCudaErrors(cudaMemcpy(&alpha, &d_s[j], sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&beta, &d_deltaS[j], sizeof(double), cudaMemcpyDeviceToHost));
		if (beta < 0.0)
		{
			alpha = -(alpha / beta);
			*alphaMaxDual = *alphaMaxDual < alpha ? *alphaMaxDual : alpha;
		}
	}
}
