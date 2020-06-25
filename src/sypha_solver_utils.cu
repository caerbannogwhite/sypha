#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}


/*
#define DEBUG_SYNC __syncthreads();

__device__ void warp_reduce_max(float smem[64])
{

	smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ? 
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ? 
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ? 
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ? 
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ? 
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;

	smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ? 
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;

}*/

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void array_mult_dev(double *A, double *B, double *C) {
	int i = threadIdx.x;
	C[i] = A[i] * B[i];
}

void array_mult_host(double *A, double *B, double *C, const int N) {
	for (int idx = 0; idx < N; idx++) {
		C[idx] = A[idx] * B[idx];
	}
}

void array_mult_host_test(double *d_A, double *d_B, double *d_C, const int N) {
	double alpha, beta;
	for (int j = 0; j < N; ++j)
    {
        cudaMemcpy(&alpha, &d_A[j], sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&beta, &d_B[j], sizeof(double), cudaMemcpyDeviceToHost);
        alpha = - (alpha * beta);
        cudaMemcpy(&d_C[j], &alpha, sizeof(double), cudaMemcpyHostToDevice);
    }
}

void initialData(double *ip,int size) {
	// generate different seed for random number
	time_t t;
	srand((unsigned int) time(&t));
	for (int i=0; i<size; i++) {
		ip[i] = (double)( rand() & 0xFF )/10.0;
	}
}

bool checkResult(double *h_res, double *d_res, int n)
{
	double buff;
	for (int i = 0; i < n; ++i)
	{
		cudaMemcpy(&buff, &d_res[i], sizeof(double), cudaMemcpyDeviceToHost);
		if (abs(buff - h_res[i]) >= 1.E-12)
		{
			printf("Test failed on i=%d, host: %lf, dev: %lf\n", i, h_res[i], buff);
			return false;
		}
		//printf("%lf, %lf\n", h_res[i], buff);
	}
	printf("Test pass\n");
	return true;
}

int main(int argc, char **argv) {
	int repeat = 20;
	int nElem = 32*32;
	size_t nBytes = nElem * sizeof(double);
	
	double tStart, tEnd, sum;

	double *h_A, *h_B, *h_C;
	double *d_A, *d_B, *d_C;

	h_A = (double *)malloc(nBytes);
	h_B = (double *)malloc(nBytes);
	h_C = (double *)malloc(nBytes);

	cudaMalloc((void **)&d_A, nBytes);
	cudaMalloc((void **)&d_B, nBytes);
	cudaMalloc((void **)&d_C, nBytes);
	
	std::cout << "Initializing data" << std::endl;
	initialData(h_A, nElem);
	initialData(h_B, nElem);

	sum = 0;
	for (int i = 0; i < 5; ++i)
	{
		tStart = cpuSecond();
		array_mult_host(h_A, h_B, h_C, nElem);
		tEnd = cpuSecond();
		sum += (tEnd - tStart) * 1000;
	}
	std::cout << "array_mult_host time " << (sum / repeat) << " ms" << std::endl;

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	
	// sum = 0;
	// for (int i = 0; i < repeat; ++i)
	// {
	// 	tStart = cpuSecond();
	// 	array_mult_host_test(h_A, h_B, h_C, nElem);
	// 	tEnd = cpuSecond();
	// 	sum += (tEnd - tStart) * 1000;
	// }
	// std::cout << "array_mult_host_test time " << (sum / repeat) << " ms" << std::endl; 
	
	sum = 0;
	for (int i = 0; i < repeat; ++i)
	{
		tStart = cpuSecond();
		array_mult_dev<<<1, 32>>>(d_A, d_B, d_C);
		cudaDeviceSynchronize();
		tEnd = cpuSecond();
		sum += (tEnd - tStart) * 1000;
	}
	std::cout << "array_mult_dev time " << (sum / repeat) << " ms" << std::endl; 
	

	checkResult(h_C, d_C, nElem);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return(0);
}
