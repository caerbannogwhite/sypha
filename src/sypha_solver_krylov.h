#ifndef SYPHA_SOLVER_KRYLOV_H
#define SYPHA_SOLVER_KRYLOV_H

#include <cstddef>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cublas_v2.h"

struct KrylovSolveWorkspace
{
    bool isAllocated = false;
    int capacityM = 0;
    int capacityN = 0;

    // CG vectors (size m on device)
    double *d_r = NULL;
    double *d_z = NULL;
    double *d_p = NULL;
    double *d_Ap = NULL;

    // Intermediate (size n on device) for two-step SpMV: A^T*p then D^2.*q then A*q
    double *d_tmp_n = NULL;

    // Normal-equations RHS (size m)
    double *d_ne_rhs = NULL;

    // D^2 = x/s (size n) and Jacobi diagonal (size m)
    double *d_D2 = NULL;
    double *d_precond_diag = NULL;

    // cuSPARSE SpMV buffer
    void *d_spmvBuffer = NULL;
    size_t spmvBufferCapacity = 0;

    // CG tolerance parameters (adaptive)
    int maxCgIter = 500;
    double cgTolInitial = 1e-2;
    double cgTolFinal = 1e-8;
    double cgTolDecayRate = 0.5;
};

void initializeKrylovWorkspace(KrylovSolveWorkspace *ws, int m, int n);
void releaseKrylovWorkspace(KrylovSolveWorkspace *ws);

void krylovComputeD2(KrylovSolveWorkspace *ws,
                     const double *d_x, const double *d_s,
                     int n, cudaStream_t stream);

void krylovComputeJacobiDiag(KrylovSolveWorkspace *ws,
                             const int *d_csrOffs, const int *d_csrInds,
                             const double *d_csrVals,
                             int m, int n, cudaStream_t stream);

void krylovBuildNormalEquationsRHS(KrylovSolveWorkspace *ws,
                                   const double *d_resC, const double *d_resB,
                                   const double *d_resXS,
                                   const double *d_x, const double *d_s,
                                   int m, int n,
                                   cusparseSpMatDescr_t matDescr,
                                   cusparseHandle_t cusparseHandle,
                                   cudaStream_t stream);

int krylovSolveCG(KrylovSolveWorkspace *ws,
                  double *d_dy,
                  const double *d_rhs,
                  double tol,
                  int m, int n,
                  cusparseSpMatDescr_t matDescr,
                  cusparseHandle_t cusparseHandle,
                  cublasHandle_t cublasHandle,
                  cudaStream_t stream);

void krylovRecoverDxDs(KrylovSolveWorkspace *ws,
                       double *d_dx, double *d_ds, const double *d_dy,
                       const double *d_resC, const double *d_resXS,
                       const double *d_x, const double *d_s,
                       int m, int n,
                       cusparseSpMatDescr_t matDescr,
                       cusparseHandle_t cusparseHandle,
                       cudaStream_t stream);

#endif // SYPHA_SOLVER_KRYLOV_H
