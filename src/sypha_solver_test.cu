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

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void array_mult_kernel(double *A, double *B, double *C)
{
	C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * 
											   B[blockIdx.x * blockDim.x + threadIdx.x];
}

void array_mult_dev(double *d_A, double *d_B, double *d_C, int N)
{
	int bSize;
	int tSize = 32;
	
	bSize = (N >> 5) + 1;
	
	//while (res > 128)
	{
		//bSize = res >> 5; // 32 = 2 ^ 5
		array_mult_kernel<<<bSize, tSize>>>(d_A, d_B, d_C);
	}
	cudaDeviceSynchronize();
}

void array_mult_host_naive(double *A, double *B, double *C, const int N)
{
	for (int idx = 0; idx < N; idx++)
	{
		C[idx] = A[idx] * B[idx];
	}
}

void array_mult_host_test_2(double *d_A, double *d_B, double *d_C, const int N)
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
		buffC[j] = buffA[j] * buffB[j];
	}

	cudaMemcpy(d_C, buffC, sizeof(double) * N, cudaMemcpyHostToDevice);

	free(buffA);
	free(buffB);
	free(buffC);
}

void array_mult_host_test_3(double *h_A, double *h_B, double *h_C,
							double *d_A, double *d_B, double *d_C,
							const int N)
{	
	cudaMemcpy(h_A, d_A, sizeof(double) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);
	
	for (int j = 0; j < N; ++j)
    {
		h_C[j] = h_A[j] * h_B[j];
	}

	cudaMemcpy(d_C, h_C, sizeof(double) * N, cudaMemcpyHostToDevice);
}

void array_mult_host_hybr(double *d_A, double *d_B, double *d_C, int N)
{
	int off = 0;
	if (N >= 1024)
	{
		int bSize = N / 1024;
		off = bSize * 1024;
		N -= off;
		array_mult_kernel<<<bSize, 32>>>(d_A, d_B, d_C);
	}
	
	array_mult_host_test_2(&d_A[off], &d_B[off], &d_C[off], N);
	cudaDeviceSynchronize();
}

void initialData(double *ip,int size)
{
	// generate different seed for random number
	time_t t;
	srand((unsigned int) time(&t));
	for (int i=0; i<size; i++)
	{
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

void launchTest(int N, int repeat)
{
	double tStart, tEnd, sum;

	double *h_A, *h_B, *h_C;
	double *d_A, *d_B, *d_C;

	h_A = (double *)malloc(sizeof(double) * N);
	h_B = (double *)malloc(sizeof(double) * N);
	h_C = (double *)malloc(sizeof(double) * N);

	cudaMalloc((void **)&d_A, sizeof(double) * N);
	cudaMalloc((void **)&d_B, sizeof(double) * N);
	cudaMalloc((void **)&d_C, sizeof(double) * N);
	
	printf("\nInitializing data (size=%d)\n", N);
	initialData(h_A, N);
	initialData(h_B, N);
	
	cudaMemcpy(d_A, h_A, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(double) * N, cudaMemcpyHostToDevice);

	// sum = 0;
	// for (int i = 0; i < repeat; ++i)
	// {
	// 	tStart = cpuSecond();
	// 	array_mult_host_naive(h_A, h_B, h_C, N);
	// 	tEnd = cpuSecond();
	// 	sum += (tEnd - tStart);
	// }
	// printf("%30s - %12.9lf ms\n", "array_mult_host_naive", (sum / repeat * 1000));

	// sum = 0;
	// for (int i = 0; i < repeat; ++i)
	// {
	// 	tStart = cpuSecond();
	// 	array_mult_host_test_3(h_A, h_B, h_C, d_A, d_B, d_C, N);
	// 	tEnd = cpuSecond();
	// 	sum += (tEnd - tStart);
	// }
	// printf("%30s - %12.9lf ms\n", "array_mult_host_test_3", (sum / repeat * 1000));

	sum = 0;
	for (int i = 0; i < repeat; ++i)
	{
		tStart = cpuSecond();
		array_mult_host_test_2(d_A, d_B, d_C, N);
		tEnd = cpuSecond();
		sum += (tEnd - tStart);
	}
	printf("%30s - %12.9lf ms\n", "array_mult_host_test_2", (sum / repeat * 1000));
	
	sum = 0;
	for (int i = 0; i < repeat; ++i)
	{
		tStart = cpuSecond();
		array_mult_host_hybr(d_A, d_B, d_C, N);
		tEnd = cpuSecond();
		sum += (tEnd - tStart);
	}
	printf("%30s - %12.9lf ms\n", "array_mult_host_hybr", (sum / repeat * 1000));
	
	array_mult_host_hybr(d_A, d_B, d_C, N);
	array_mult_host_naive(h_A, h_B, h_C, N);
	checkResult(h_C, d_C, N);
	
	// sum = 0;
	// for (int i = 0; i < repeat; ++i)
	// {
	// 	tStart = cpuSecond();
	// 	array_mult_dev(d_A, d_B, d_C, N);
	// 	tEnd = cpuSecond();
	// 	sum += (tEnd - tStart);
	// }
	// printf("%30s - %12.9lf ms\n", "array_mult_dev", (sum / repeat * 1000));
	// array_mult_host_naive(h_A, h_B, h_C, N);
	// checkResult(h_C, d_C, N);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
}

int main(int argc, char **argv)
{
	for (int i = atoi(argv[1]); i < atoi(argv[2]); i *= 2)
	{
		launchTest(i, 10);
	}
	
	return(0);
}
