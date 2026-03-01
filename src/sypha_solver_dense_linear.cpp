#include "sypha_solver.h"
#include "sypha_node_sparse.h"
#include "sypha_cuda_helper.h"

void initializeDenseLinearSolveWorkspace(DenseLinearSolveWorkspace *workspace,
                                         int nRows,
                                         int nnz,
                                         int *dCsrRowOffsets,
                                         int *dCsrColIndices,
                                         double *dCsrValues,
                                         cusparseHandle_t cusparseHandle,
                                         cusolverDnHandle_t cusolverDnHandle)
{
    workspace->isEnabled = true;
    workspace->nRows = nRows;

    checkCudaErrors(cusparseCreateCsr(&workspace->sparseMatrix,
                                      static_cast<int64_t>(nRows), static_cast<int64_t>(nRows), static_cast<int64_t>(nnz),
                                      dCsrRowOffsets, dCsrColIndices, dCsrValues,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->dDenseA), sizeof(double) * static_cast<size_t>(nRows) * static_cast<size_t>(nRows)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->dDenseA_template), sizeof(double) * static_cast<size_t>(nRows) * static_cast<size_t>(nRows)));
    checkCudaErrors(cusparseCreateDnMat(&workspace->denseMatrix,
                                        static_cast<int64_t>(nRows), static_cast<int64_t>(nRows), static_cast<int64_t>(nRows),
                                        workspace->dDenseA, CUDA_R_64F, CUSPARSE_ORDER_COL));

    checkCudaErrors(cusparseSparseToDense_bufferSize(cusparseHandle, workspace->sparseMatrix, workspace->denseMatrix,
                                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &workspace->sparseToDenseBufferSize));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->dSparseToDenseBuffer), workspace->sparseToDenseBufferSize));

    checkCudaErrors(cusolverDnDgetrf_bufferSize(cusolverDnHandle,
                                                nRows, nRows,
                                                workspace->dDenseA, nRows,
                                                &workspace->luWorkSize));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->dLuWork), sizeof(double) * static_cast<size_t>(workspace->luWorkSize)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->dLuPivot), sizeof(int) * static_cast<size_t>(nRows)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->dLuInfo), sizeof(int)));
}

void releaseDenseLinearSolveWorkspace(DenseLinearSolveWorkspace *workspace)
{
    if (workspace->sparseMatrix)
    {
        checkCudaErrors(cusparseDestroySpMat(workspace->sparseMatrix));
        workspace->sparseMatrix = nullptr;
    }
    if (workspace->denseMatrix)
    {
        checkCudaErrors(cusparseDestroyDnMat(workspace->denseMatrix));
        workspace->denseMatrix = nullptr;
    }
    if (workspace->dDenseA)
    {
        checkCudaErrors(cudaFree(workspace->dDenseA));
        workspace->dDenseA = nullptr;
    }
    if (workspace->dDenseA_template)
    {
        checkCudaErrors(cudaFree(workspace->dDenseA_template));
        workspace->dDenseA_template = nullptr;
    }
    if (workspace->dLuWork)
    {
        checkCudaErrors(cudaFree(workspace->dLuWork));
        workspace->dLuWork = nullptr;
    }
    if (workspace->dLuPivot)
    {
        checkCudaErrors(cudaFree(workspace->dLuPivot));
        workspace->dLuPivot = nullptr;
    }
    if (workspace->dLuInfo)
    {
        checkCudaErrors(cudaFree(workspace->dLuInfo));
        workspace->dLuInfo = nullptr;
    }
    if (workspace->dSparseToDenseBuffer)
    {
        checkCudaErrors(cudaFree(workspace->dSparseToDenseBuffer));
        workspace->dSparseToDenseBuffer = nullptr;
    }
    workspace->isEnabled = false;
}

bool solveDenseLinearSystem(DenseLinearSolveWorkspace *workspace,
                            const double *dRhs,
                            double *dSolution,
                            cusparseHandle_t cusparseHandle,
                            cusolverDnHandle_t cusolverDnHandle,
                            cudaStream_t cudaStream)
{
    int info = 0;
    checkCudaErrors(cusparseSparseToDense(cusparseHandle, workspace->sparseMatrix, workspace->denseMatrix,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT, workspace->dSparseToDenseBuffer));
    checkCudaErrors(cudaMemcpyAsync(dSolution, dRhs, sizeof(double) * static_cast<size_t>(workspace->nRows),
                                    cudaMemcpyDeviceToDevice, cudaStream));
    checkCudaErrors(cusolverDnDgetrf(cusolverDnHandle,
                                     workspace->nRows, workspace->nRows,
                                     workspace->dDenseA, workspace->nRows,
                                     workspace->dLuWork, workspace->dLuPivot, workspace->dLuInfo));
    checkCudaErrors(cudaMemcpy(&info, workspace->dLuInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
    {
        return false;
    }
    checkCudaErrors(cusolverDnDgetrs(cusolverDnHandle, CUBLAS_OP_N,
                                     workspace->nRows, 1,
                                     workspace->dDenseA, workspace->nRows,
                                     workspace->dLuPivot, dSolution, workspace->nRows,
                                     workspace->dLuInfo));
    // Dgetrs triangular solve: skip synchronous D->H info check.
    return true;
}

bool factorizeDenseLinearSystem(DenseLinearSolveWorkspace *workspace,
                                cusparseHandle_t cusparseHandle,
                                cusolverDnHandle_t cusolverDnHandle)
{
    int info = 0;
    checkCudaErrors(cusparseSparseToDense(cusparseHandle, workspace->sparseMatrix, workspace->denseMatrix,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT, workspace->dSparseToDenseBuffer));
    checkCudaErrors(cusolverDnDgetrf(cusolverDnHandle,
                                     workspace->nRows, workspace->nRows,
                                     workspace->dDenseA, workspace->nRows,
                                     workspace->dLuWork, workspace->dLuPivot, workspace->dLuInfo));
    checkCudaErrors(cudaMemcpy(&info, workspace->dLuInfo, sizeof(int), cudaMemcpyDeviceToHost));
    return info == 0;
}

void buildDenseKktTemplate(DenseLinearSolveWorkspace *workspace,
                            int ncols, int nrows,
                            cusparseHandle_t cusparseHandle)
{
    // Full sparse-to-dense conversion to capture the static KKT structure.
    checkCudaErrors(cusparseSparseToDense(cusparseHandle, workspace->sparseMatrix, workspace->denseMatrix,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT, workspace->dSparseToDenseBuffer));
    // Save as template (LU factorization will destroy dDenseA in-place).
    const size_t matBytes = sizeof(double) * static_cast<size_t>(workspace->nRows) * static_cast<size_t>(workspace->nRows);
    checkCudaErrors(cudaMemcpy(workspace->dDenseA_template, workspace->dDenseA, matBytes, cudaMemcpyDeviceToDevice));

    // Precompute diagonal offsets in column-major layout.
    // KKT rows: [0..ncols-1] = x-block, [ncols..ncols+nrows-1] = y-block,
    //           [ncols+nrows..2*ncols+nrows-1] = s-block (contains S and X diags).
    // S diagonal: (ncols+nrows+j, j)        → offset = (ncols+nrows) + j*(nRows+1)
    // X diagonal: (ncols+nrows+j, ncols+nrows+j) → offset = (ncols+nrows)*(nRows+1) + j*(nRows+1)
    const int n = workspace->nRows;
    workspace->diagSOffset = ncols + nrows;                    // start of S diagonal
    workspace->diagXOffset = (ncols + nrows) * (n + 1);       // start of X diagonal
    workspace->diagStride = n + 1;                              // diagonal stride in col-major
    workspace->nDiag = ncols;
    workspace->templateReady = true;
}

bool factorizeDenseLinearSystemIncremental(DenseLinearSolveWorkspace *workspace,
                                            const double *d_s, const double *d_x,
                                            cublasHandle_t cublasHandle,
                                            cusolverDnHandle_t cusolverDnHandle,
                                            cudaStream_t cudaStream)
{
    // Restore template (fast contiguous D2D copy) — undoes LU destruction from prev iteration.
    const size_t matBytes = sizeof(double) * static_cast<size_t>(workspace->nRows) * static_cast<size_t>(workspace->nRows);
    checkCudaErrors(cudaMemcpyAsync(workspace->dDenseA, workspace->dDenseA_template,
                                     matBytes, cudaMemcpyDeviceToDevice, cudaStream));

    // Patch S diagonal: dDenseA[(ncols+nrows+j) + j*nRows] for j=0..ncols-1
    checkCudaErrors(cublasDcopy(cublasHandle, workspace->nDiag,
                                d_s, 1,
                                &workspace->dDenseA[workspace->diagSOffset], workspace->diagStride));

    // Patch X diagonal: dDenseA[(ncols+nrows+j) + (ncols+nrows+j)*nRows]
    checkCudaErrors(cublasDcopy(cublasHandle, workspace->nDiag,
                                d_x, 1,
                                &workspace->dDenseA[workspace->diagXOffset], workspace->diagStride));

    // LU factorize (in-place, destroys dDenseA).
    int info = 0;
    checkCudaErrors(cusolverDnDgetrf(cusolverDnHandle,
                                     workspace->nRows, workspace->nRows,
                                     workspace->dDenseA, workspace->nRows,
                                     workspace->dLuWork, workspace->dLuPivot, workspace->dLuInfo));
    checkCudaErrors(cudaMemcpy(&info, workspace->dLuInfo, sizeof(int), cudaMemcpyDeviceToHost));
    return info == 0;
}

bool solveDenseLinearSystemFactored(DenseLinearSolveWorkspace *workspace,
                                     const double *dRhs,
                                     double *dSolution,
                                     cusolverDnHandle_t cusolverDnHandle,
                                     cudaStream_t cudaStream)
{
    checkCudaErrors(cudaMemcpyAsync(dSolution, dRhs, sizeof(double) * static_cast<size_t>(workspace->nRows),
                                    cudaMemcpyDeviceToDevice, cudaStream));
    checkCudaErrors(cusolverDnDgetrs(cusolverDnHandle, CUBLAS_OP_N,
                                     workspace->nRows, 1,
                                     workspace->dDenseA, workspace->nRows,
                                     workspace->dLuPivot, dSolution, workspace->nRows,
                                     workspace->dLuInfo));
    // Dgetrs is a triangular solve; if LU factorization succeeded, it cannot fail.
    // Skip synchronous D->H info check to avoid pipeline stall.
    return true;
}
