#ifndef SYPHA_SOLVER_UTILS_H
#define SYPHA_SOLVER_UTILS_H

#include "common.h"
#include <cuda_runtime.h>

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, int N, cudaStream_t stream = 0);
void elem_min_mult_host(double *d_A, double *d_B, double *d_C, const int N);
void elem_min_mult_hybr(double *d_A, double *d_B, double *d_C, int N);

/** Corrector RHS: out[j] = -deltaX[j]*deltaS[j] + sigma*mu. All on device. */
void corrector_rhs_dev(double *d_deltaX, double *d_deltaS, double sigma, double mu,
                       double *d_out, int N, cudaStream_t stream = 0);

/** Alpha-max step length on device: alphaMaxPrim = min(-x/dx for dx<0), alphaMaxDual = min(-s/ds for ds<0).
 *  Caller provides d_tmp_prim, d_tmp_dual (length N), d_blockmin_prim, d_blockmin_dual (length ceil(N/256)),
 *  d_alphaResult (2 doubles on device for final reduction result). */
void alpha_max_dev(const double *d_x, const double *d_deltaX, const double *d_s, const double *d_deltaS,
                   int N,
                   double *d_tmp_prim, double *d_tmp_dual,
                   double *d_blockmin_prim, double *d_blockmin_dual,
                   double *d_alphaResult,
                   double *alphaMaxPrim, double *alphaMaxDual,
                   cudaStream_t stream = 0);

#endif // SYPHA_SOLVER_UTILS_H
