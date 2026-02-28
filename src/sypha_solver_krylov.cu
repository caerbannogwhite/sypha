#include "sypha_solver_krylov.h"
#include "sypha_cuda_helper.h"

#include <cfloat>
#include <cmath>
#include <cstdio>

// CUDA 12+ cuSPARSE: legacy enums removed; use generic API enums
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
#ifndef CUSPARSE_CSRMV_ALG2
#define CUSPARSE_CSRMV_ALG2 CUSPARSE_SPMV_CSR_ALG2
#endif
#endif

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

__global__ void compute_D2_kernel(double *D2, const double *x, const double *s, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
        D2[j] = x[j] / s[j];
}

__global__ void jacobi_diag_kernel(double *diag,
                                   const int *csrOffs, const int *csrInds,
                                   const double *csrVals, const double *D2,
                                   int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;
    double sum = 0.0;
    int start = csrOffs[i];
    int end = csrOffs[i + 1];
    for (int k = start; k < end; ++k)
    {
        double v = csrVals[k];
        sum += v * v * D2[csrInds[k]];
    }
    diag[i] = sum;
}

__global__ void jacobi_precondition_kernel(double *z, const double *r,
                                           const double *diag, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
        z[i] = r[i] / fmax(diag[i], 1e-30);
}

// tmp[j] = (x[j]*resC[j] - resXS[j]) / s[j]
// Note: resXS = -x.*s from elem_min_mult_dev, so x*resC - resXS = x*resC + x*s
__global__ void build_ne_rhs_tmp_kernel(double *tmp,
                                        const double *resC, const double *resXS,
                                        const double *x, const double *s,
                                        int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
        tmp[j] = (x[j] * resC[j] - resXS[j]) / s[j];
}

__global__ void scale_by_D2_kernel(double *out, const double *in,
                                   const double *D2, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
        out[j] = in[j] * D2[j];
}

// dx[j] = (resXS[j] - x[j]*ds[j]) / s[j]
__global__ void recover_dx_kernel(double *dx,
                                  const double *ds, const double *resXS,
                                  const double *x, const double *s,
                                  int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
        dx[j] = (resXS[j] - x[j] * ds[j]) / s[j];
}

// ---------------------------------------------------------------------------
// Workspace management (grow-only allocation)
// ---------------------------------------------------------------------------

void initializeKrylovWorkspace(KrylovSolveWorkspace *ws, int m, int n)
{
    if (m <= ws->capacityM && n <= ws->capacityN && ws->isAllocated)
        return;

    // Free old allocations if growing
    if (ws->d_r) { checkCudaErrors(cudaFree(ws->d_r)); ws->d_r = NULL; }
    if (ws->d_z) { checkCudaErrors(cudaFree(ws->d_z)); ws->d_z = NULL; }
    if (ws->d_p) { checkCudaErrors(cudaFree(ws->d_p)); ws->d_p = NULL; }
    if (ws->d_Ap) { checkCudaErrors(cudaFree(ws->d_Ap)); ws->d_Ap = NULL; }
    if (ws->d_ne_rhs) { checkCudaErrors(cudaFree(ws->d_ne_rhs)); ws->d_ne_rhs = NULL; }
    if (ws->d_precond_diag) { checkCudaErrors(cudaFree(ws->d_precond_diag)); ws->d_precond_diag = NULL; }
    if (ws->d_tmp_n) { checkCudaErrors(cudaFree(ws->d_tmp_n)); ws->d_tmp_n = NULL; }
    if (ws->d_D2) { checkCudaErrors(cudaFree(ws->d_D2)); ws->d_D2 = NULL; }

    int allocM = (m > ws->capacityM) ? m : ws->capacityM;
    int allocN = (n > ws->capacityN) ? n : ws->capacityN;

    checkCudaErrors(cudaMalloc((void **)&ws->d_r, sizeof(double) * (size_t)allocM));
    checkCudaErrors(cudaMalloc((void **)&ws->d_z, sizeof(double) * (size_t)allocM));
    checkCudaErrors(cudaMalloc((void **)&ws->d_p, sizeof(double) * (size_t)allocM));
    checkCudaErrors(cudaMalloc((void **)&ws->d_Ap, sizeof(double) * (size_t)allocM));
    checkCudaErrors(cudaMalloc((void **)&ws->d_ne_rhs, sizeof(double) * (size_t)allocM));
    checkCudaErrors(cudaMalloc((void **)&ws->d_precond_diag, sizeof(double) * (size_t)allocM));

    checkCudaErrors(cudaMalloc((void **)&ws->d_tmp_n, sizeof(double) * (size_t)allocN));
    checkCudaErrors(cudaMalloc((void **)&ws->d_D2, sizeof(double) * (size_t)allocN));

    ws->capacityM = allocM;
    ws->capacityN = allocN;
    ws->isAllocated = true;
}

void releaseKrylovWorkspace(KrylovSolveWorkspace *ws)
{
    if (ws->d_r) { checkCudaErrors(cudaFree(ws->d_r)); ws->d_r = NULL; }
    if (ws->d_z) { checkCudaErrors(cudaFree(ws->d_z)); ws->d_z = NULL; }
    if (ws->d_p) { checkCudaErrors(cudaFree(ws->d_p)); ws->d_p = NULL; }
    if (ws->d_Ap) { checkCudaErrors(cudaFree(ws->d_Ap)); ws->d_Ap = NULL; }
    if (ws->d_ne_rhs) { checkCudaErrors(cudaFree(ws->d_ne_rhs)); ws->d_ne_rhs = NULL; }
    if (ws->d_precond_diag) { checkCudaErrors(cudaFree(ws->d_precond_diag)); ws->d_precond_diag = NULL; }
    if (ws->d_tmp_n) { checkCudaErrors(cudaFree(ws->d_tmp_n)); ws->d_tmp_n = NULL; }
    if (ws->d_D2) { checkCudaErrors(cudaFree(ws->d_D2)); ws->d_D2 = NULL; }
    if (ws->d_spmvBuffer) { checkCudaErrors(cudaFree(ws->d_spmvBuffer)); ws->d_spmvBuffer = NULL; }
    ws->spmvBufferCapacity = 0;
    ws->capacityM = 0;
    ws->capacityN = 0;
    ws->isAllocated = false;
}

// ---------------------------------------------------------------------------
// D^2 and Jacobi preconditioner
// ---------------------------------------------------------------------------

void krylovComputeD2(KrylovSolveWorkspace *ws,
                     const double *d_x, const double *d_s,
                     int n, cudaStream_t stream)
{
    const int blockDim = 256;
    const int gridDim = (n + blockDim - 1) / blockDim;
    compute_D2_kernel<<<gridDim, blockDim, 0, stream>>>(ws->d_D2, d_x, d_s, n);
}

void krylovComputeJacobiDiag(KrylovSolveWorkspace *ws,
                             const int *d_csrOffs, const int *d_csrInds,
                             const double *d_csrVals,
                             int m, int n, cudaStream_t stream)
{
    const int blockDim = 256;
    const int gridDim = (m + blockDim - 1) / blockDim;
    jacobi_diag_kernel<<<gridDim, blockDim, 0, stream>>>(
        ws->d_precond_diag, d_csrOffs, d_csrInds, d_csrVals, ws->d_D2, m);
}

// ---------------------------------------------------------------------------
// SpMV buffer helper (grow-only)
// ---------------------------------------------------------------------------

static void ensureSpmvBuffer(KrylovSolveWorkspace *ws, size_t needed)
{
    if (needed <= ws->spmvBufferCapacity)
        return;
    if (ws->d_spmvBuffer)
        checkCudaErrors(cudaFree(ws->d_spmvBuffer));
    checkCudaErrors(cudaMalloc(&ws->d_spmvBuffer, needed));
    ws->spmvBufferCapacity = needed;
}

// ---------------------------------------------------------------------------
// RHS construction: f = resB - A * S^{-1}(X*resC - resXS)
// ---------------------------------------------------------------------------

void krylovBuildNormalEquationsRHS(KrylovSolveWorkspace *ws,
                                   const double *d_resC, const double *d_resB,
                                   const double *d_resXS,
                                   const double *d_x, const double *d_s,
                                   int m, int n,
                                   cusparseSpMatDescr_t matDescr,
                                   cusparseHandle_t cusparseHandle,
                                   cudaStream_t stream)
{
    // Step 1: tmp_n[j] = (x[j]*resC[j] - resXS[j]) / s[j]
    {
        const int blockDim = 256;
        const int gridDim = (n + blockDim - 1) / blockDim;
        build_ne_rhs_tmp_kernel<<<gridDim, blockDim, 0, stream>>>(
            ws->d_tmp_n, d_resC, d_resXS, d_x, d_s, n);
    }

    // Step 2: ne_rhs = resB
    checkCudaErrors(cudaMemcpyAsync(ws->d_ne_rhs, d_resB,
                                    sizeof(double) * (size_t)m,
                                    cudaMemcpyDeviceToDevice, stream));

    // Step 3: ne_rhs -= A * tmp_n  (i.e. alpha=-1, beta=1)
    cusparseDnVecDescr_t vecTmpN, vecRhs;
    checkCudaErrors(cusparseCreateDnVec(&vecTmpN, (int64_t)n, ws->d_tmp_n, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecRhs, (int64_t)m, ws->d_ne_rhs, CUDA_R_64F));

    double alpha = -1.0, beta = 1.0;
    size_t bufSize = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matDescr, vecTmpN,
                                            &beta, vecRhs, CUDA_R_64F,
                                            CUSPARSE_CSRMV_ALG2, &bufSize));
    ensureSpmvBuffer(ws, bufSize);
    checkCudaErrors(cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matDescr, vecTmpN,
                                 &beta, vecRhs, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG2, ws->d_spmvBuffer));

    checkCudaErrors(cusparseDestroyDnVec(vecTmpN));
    checkCudaErrors(cusparseDestroyDnVec(vecRhs));
}

// ---------------------------------------------------------------------------
// Preconditioned CG solver for A D^2 A^T dy = rhs
// Returns iteration count on success, -1 on failure.
// ---------------------------------------------------------------------------

int krylovSolveCG(KrylovSolveWorkspace *ws,
                  double *d_dy,
                  const double *d_rhs,
                  double tol,
                  int m, int n,
                  cusparseSpMatDescr_t matDescr,
                  cusparseHandle_t cusparseHandle,
                  cublasHandle_t cublasHandle,
                  cudaStream_t stream)
{
    const int blockDim = 256;

    // Compute ||rhs||_2 for relative convergence check
    double rhsNorm = 0.0;
    checkCudaErrors(cublasDnrm2(cublasHandle, m, d_rhs, 1, &rhsNorm));
    if (rhsNorm < 1e-30)
    {
        // RHS is zero => solution is zero
        checkCudaErrors(cudaMemsetAsync(d_dy, 0, sizeof(double) * (size_t)m, stream));
        return 0;
    }

    // Create dense vector descriptors for SpMV
    cusparseDnVecDescr_t vecP, vecTmpN, vecAp;
    checkCudaErrors(cusparseCreateDnVec(&vecP, (int64_t)m, ws->d_p, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecTmpN, (int64_t)n, ws->d_tmp_n, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecAp, (int64_t)m, ws->d_Ap, CUDA_R_64F));

    // Ensure SpMV buffer is large enough for both transpose and non-transpose
    {
        double one = 1.0, zero = 0.0;
        size_t bufSize1 = 0, bufSize2 = 0;
        checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle,
                                                CUSPARSE_OPERATION_TRANSPOSE,
                                                &one, matDescr, vecP,
                                                &zero, vecTmpN, CUDA_R_64F,
                                                CUSPARSE_CSRMV_ALG2, &bufSize1));
        checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &one, matDescr, vecTmpN,
                                                &zero, vecAp, CUDA_R_64F,
                                                CUSPARSE_CSRMV_ALG2, &bufSize2));
        size_t maxBuf = (bufSize1 > bufSize2) ? bufSize1 : bufSize2;
        ensureSpmvBuffer(ws, maxBuf);
    }

    // Initial guess: dy = 0
    checkCudaErrors(cudaMemsetAsync(d_dy, 0, sizeof(double) * (size_t)m, stream));

    // r = rhs (since A D^2 A^T * 0 = 0)
    checkCudaErrors(cudaMemcpyAsync(ws->d_r, d_rhs,
                                    sizeof(double) * (size_t)m,
                                    cudaMemcpyDeviceToDevice, stream));

    // z = M^{-1} r (Jacobi preconditioner)
    {
        int gridDim = (m + blockDim - 1) / blockDim;
        jacobi_precondition_kernel<<<gridDim, blockDim, 0, stream>>>(
            ws->d_z, ws->d_r, ws->d_precond_diag, m);
    }

    // p = z
    checkCudaErrors(cudaMemcpyAsync(ws->d_p, ws->d_z,
                                    sizeof(double) * (size_t)m,
                                    cudaMemcpyDeviceToDevice, stream));

    // rz = dot(r, z)
    double rz = 0.0;
    checkCudaErrors(cublasDdot(cublasHandle, m, ws->d_r, 1, ws->d_z, 1, &rz));

    int maxIter = ws->maxCgIter;
    int iter = 0;

    for (iter = 0; iter < maxIter; ++iter)
    {
        // --- Matvec: Ap = A D^2 A^T p ---
        // Step 1: tmp_n = A^T * p
        {
            double one = 1.0, zero = 0.0;
            checkCudaErrors(cusparseSpMV(cusparseHandle,
                                         CUSPARSE_OPERATION_TRANSPOSE,
                                         &one, matDescr, vecP,
                                         &zero, vecTmpN, CUDA_R_64F,
                                         CUSPARSE_CSRMV_ALG2, ws->d_spmvBuffer));
        }
        // Step 2: tmp_n = D^2 .* tmp_n
        {
            int gridDim = (n + blockDim - 1) / blockDim;
            scale_by_D2_kernel<<<gridDim, blockDim, 0, stream>>>(
                ws->d_tmp_n, ws->d_tmp_n, ws->d_D2, n);
        }
        // Step 3: Ap = A * tmp_n
        {
            double one = 1.0, zero = 0.0;
            checkCudaErrors(cusparseSpMV(cusparseHandle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one, matDescr, vecTmpN,
                                         &zero, vecAp, CUDA_R_64F,
                                         CUSPARSE_CSRMV_ALG2, ws->d_spmvBuffer));
        }

        // pAp = dot(p, Ap)
        double pAp = 0.0;
        checkCudaErrors(cublasDdot(cublasHandle, m, ws->d_p, 1, ws->d_Ap, 1, &pAp));

        if (pAp <= 0.0 || !std::isfinite(pAp))
        {
            iter = -1;
            break;
        }

        double alpha_cg = rz / pAp;

        // dy += alpha * p
        checkCudaErrors(cublasDaxpy(cublasHandle, m, &alpha_cg, ws->d_p, 1, d_dy, 1));

        // r -= alpha * Ap
        double neg_alpha = -alpha_cg;
        checkCudaErrors(cublasDaxpy(cublasHandle, m, &neg_alpha, ws->d_Ap, 1, ws->d_r, 1));

        // Convergence check: ||r||_2 / ||rhs||_2 < tol
        double rNorm = 0.0;
        checkCudaErrors(cublasDnrm2(cublasHandle, m, ws->d_r, 1, &rNorm));
        if (rNorm / rhsNorm < tol)
        {
            ++iter; // count completed iteration
            break;
        }

        // z = M^{-1} r
        {
            int gridDim = (m + blockDim - 1) / blockDim;
            jacobi_precondition_kernel<<<gridDim, blockDim, 0, stream>>>(
                ws->d_z, ws->d_r, ws->d_precond_diag, m);
        }

        // rz_new = dot(r, z)
        double rz_new = 0.0;
        checkCudaErrors(cublasDdot(cublasHandle, m, ws->d_r, 1, ws->d_z, 1, &rz_new));

        if (fabs(rz) < 1e-30)
        {
            iter = -1;
            break;
        }

        double beta_cg = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        checkCudaErrors(cublasDscal(cublasHandle, m, &beta_cg, ws->d_p, 1));
        double one = 1.0;
        checkCudaErrors(cublasDaxpy(cublasHandle, m, &one, ws->d_z, 1, ws->d_p, 1));
    }

    checkCudaErrors(cusparseDestroyDnVec(vecP));
    checkCudaErrors(cusparseDestroyDnVec(vecTmpN));
    checkCudaErrors(cusparseDestroyDnVec(vecAp));

    if (iter >= maxIter)
        return -1; // did not converge

    return iter;
}

// ---------------------------------------------------------------------------
// Back-substitution: recover dx, ds from dy
//   ds = resC - A^T * dy
//   dx = (resXS - x .* ds) / s
// ---------------------------------------------------------------------------

void krylovRecoverDxDs(KrylovSolveWorkspace *ws,
                       double *d_dx, double *d_ds, const double *d_dy,
                       const double *d_resC, const double *d_resXS,
                       const double *d_x, const double *d_s,
                       int m, int n,
                       cusparseSpMatDescr_t matDescr,
                       cusparseHandle_t cusparseHandle,
                       cudaStream_t stream)
{
    // ds = resC
    checkCudaErrors(cudaMemcpyAsync(d_ds, d_resC,
                                    sizeof(double) * (size_t)n,
                                    cudaMemcpyDeviceToDevice, stream));

    // ds -= A^T * dy
    cusparseDnVecDescr_t vecDy, vecDs;
    checkCudaErrors(cusparseCreateDnVec(&vecDy, (int64_t)m, const_cast<double *>(d_dy), CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecDs, (int64_t)n, d_ds, CUDA_R_64F));

    double alpha = -1.0, beta = 1.0;
    size_t bufSize = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, matDescr, vecDy,
                                            &beta, vecDs, CUDA_R_64F,
                                            CUSPARSE_CSRMV_ALG2, &bufSize));
    ensureSpmvBuffer(ws, bufSize);
    checkCudaErrors(cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, matDescr, vecDy,
                                 &beta, vecDs, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG2, ws->d_spmvBuffer));

    checkCudaErrors(cusparseDestroyDnVec(vecDy));
    checkCudaErrors(cusparseDestroyDnVec(vecDs));

    // dx[j] = (resXS[j] - x[j]*ds[j]) / s[j]
    {
        const int blockDim = 256;
        const int gridDim = (n + blockDim - 1) / blockDim;
        recover_dx_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_dx, d_ds, d_resXS, d_x, d_s, n);
    }
}
