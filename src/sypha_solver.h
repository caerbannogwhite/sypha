#ifndef SYPHA_SOLVER_H
#define SYPHA_SOLVER_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverDn.h"

struct DenseLinearSolveWorkspace
{
    bool isEnabled = false;
    int nRows = 0;
    cusparseSpMatDescr_t sparseMatrix = NULL;
    cusparseDnMatDescr_t denseMatrix = NULL;
    double *dDenseA = NULL;
    double *dLuWork = NULL;
    int *dLuPivot = NULL;
    int *dLuInfo = NULL;
    void *dSparseToDenseBuffer = NULL;
    size_t sparseToDenseBufferSize = 0;
    int luWorkSize = 0;
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

#endif // SYPHA_SOLVER_H
