#ifndef SYPHA_SOLVER_UTILS_H
#define SYPHA_SOLVER_UTILS_H

#include "common.h"

__global__ void range_kernel(int *d_A, const int N);
__global__ void range_kernel(double *d_A, const int N);
__global__ void elem_mult_kernel(double *d_A, double *d_B, double *d_C, const int N);
__global__ void elem_min_mult_kernel(double *d_A, double *d_B, double *d_C, const int N);
__global__ void elem_inv_kernel(double *d_A, double *d_invA, const int N);

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, const int N);
void elem_min_mult_host(double *d_A, double *d_B, double *d_C, const int N);
void elem_min_mult_hybr(double *d_A, double *d_B, double *d_C, int N);

void find_alpha_max(double *alphaMaxPrim, double *alphaMaxDual,
                    double *d_x, double *d_deltaX, double *d_s, double *d_deltaS, const int N);

#endif // SYPHA_SOLVER_UTILS_H