#ifndef SYPHA_SOLVER_UTILS_H
#define SYPHA_SOLVER_UTILS_H

#include "common.h"

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, int N);
void elem_min_mult_host(double *d_A, double *d_B, double *d_C, const int N);
void elem_min_mult_hybr(double *d_A, double *d_B, double *d_C, int N);

/** Corrector RHS: out[j] = -deltaX[j]*deltaS[j] + sigma*mu. All on device. */
void corrector_rhs_dev(double *d_deltaX, double *d_deltaS, double sigma, double mu,
                       double *d_out, int N);

#endif // SYPHA_SOLVER_UTILS_H