#ifndef SYPHA_SOLVER_UTILS_H
#define SYPHA_SOLVER_UTILS_H

#include "common.h"

void elem_min_mult_dev(double *d_A, double *d_B, double *d_C, int N);
void elem_min_mult_host(double *d_A, double *d_B, double *d_C, const int N);
void elem_min_mult_hybr(double *d_A, double *d_B, double *d_C, int N);

#endif // SYPHA_SOLVER_UTILS_H