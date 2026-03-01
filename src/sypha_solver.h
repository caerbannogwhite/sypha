#ifndef SYPHA_SOLVER_H
#define SYPHA_SOLVER_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverDn.h"
#include "cublas_v2.h"

struct KrylovSolveWorkspace;

struct DenseLinearSolveWorkspace
{
    bool isEnabled = false;
    int nRows = 0;
    cusparseSpMatDescr_t sparseMatrix = nullptr;
    cusparseDnMatDescr_t denseMatrix = nullptr;
    double *dDenseA = nullptr;
    double *dDenseA_template = nullptr; // Static KKT template for incremental update
    double *dLuWork = nullptr;
    int *dLuPivot = nullptr;
    int *dLuInfo = nullptr;
    void *dSparseToDenseBuffer = nullptr;
    size_t sparseToDenseBufferSize = 0;
    int luWorkSize = 0;
    bool templateReady = false;
    int diagSOffset = 0;  // Column-major offset of S diagonal start in dense matrix
    int diagXOffset = 0;  // Column-major offset of X diagonal start in dense matrix
    int diagStride = 0;   // Stride between consecutive diagonal elements (nRows + 1)
    int nDiag = 0;        // Number of diagonal elements to patch (ncols)
};

void initializeDenseLinearSolveWorkspace(DenseLinearSolveWorkspace *workspace,
                                         int nRows,
                                         int nnz,
                                         int *dCsrRowOffsets,
                                         int *dCsrColIndices,
                                         double *dCsrValues,
                                         cusparseHandle_t cusparseHandle,
                                         cusolverDnHandle_t cusolverDnHandle);

void releaseDenseLinearSolveWorkspace(DenseLinearSolveWorkspace *workspace);

bool solveDenseLinearSystem(DenseLinearSolveWorkspace *workspace,
                            const double *dRhs,
                            double *dSolution,
                            cusparseHandle_t cusparseHandle,
                            cusolverDnHandle_t cusolverDnHandle,
                            cudaStream_t cudaStream);

bool factorizeDenseLinearSystem(DenseLinearSolveWorkspace *workspace,
                                cusparseHandle_t cusparseHandle,
                                cusolverDnHandle_t cusolverDnHandle);

void buildDenseKktTemplate(DenseLinearSolveWorkspace *workspace,
                            int ncols, int nrows,
                            cusparseHandle_t cusparseHandle);

bool factorizeDenseLinearSystemIncremental(DenseLinearSolveWorkspace *workspace,
                                            const double *d_s, const double *d_x,
                                            cublasHandle_t cublasHandle,
                                            cusolverDnHandle_t cusolverDnHandle,
                                            cudaStream_t cudaStream);

bool solveDenseLinearSystemFactored(DenseLinearSolveWorkspace *workspace,
                                     const double *dRhs,
                                     double *dSolution,
                                     cusolverDnHandle_t cusolverDnHandle,
                                     cudaStream_t cudaStream);

/** Pre-allocated GPU workspace for the IPM main loop.
 *  Allocated once and reused across B&B nodes to avoid per-node cudaMalloc churn. */
struct IpmWorkspace
{
    int *d_csrAInds = nullptr;
    int *d_csrAOffs = nullptr;
    double *d_csrAVals = nullptr;
    int kktNnzCapacity = 0;
    int kktNrowsCapacity = 0;

    double *d_rhs = nullptr;
    double *d_sol = nullptr;
    double *d_prevSol = nullptr;
    int vectorCapacity = 0;

    double *d_tmp_prim = nullptr;
    double *d_tmp_dual = nullptr;
    double *d_blockmin_prim = nullptr;
    double *d_blockmin_dual = nullptr;
    double *d_alphaResult = nullptr;
    int alphaCapacity = 0;
    int alphaBlocksCapacity = 0;

    double *d_buffer = nullptr;
    size_t bufferCapacity = 0;

    cusparseMatDescr_t A_descr = nullptr;

    KrylovSolveWorkspace *krylov = nullptr;

    bool isAllocated = false;
};

void initializeIpmWorkspace(IpmWorkspace *ws, int maxKktNrows, int maxKktNnz, int maxNcols);
void releaseIpmWorkspace(IpmWorkspace *ws);

#endif // SYPHA_SOLVER_H
