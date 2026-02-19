#include "sypha_solver.h"
#include "sypha_solver_sparse.h"
#include "sypha_solver_bnb.h"
#include "sypha_solver_heuristics.h"
#include "sypha_solver_dense.h"
#include "sypha_node_sparse.h"

#include "sypha_cuda_helper.h"
#include <cstdint>

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
                                      (int64_t)nRows, (int64_t)nRows, (int64_t)nnz,
                                      dCsrRowOffsets, dCsrColIndices, dCsrValues,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    checkCudaErrors(cudaMalloc((void **)&workspace->dDenseA, sizeof(double) * (size_t)nRows * (size_t)nRows));
    checkCudaErrors(cusparseCreateDnMat(&workspace->denseMatrix,
                                        (int64_t)nRows, (int64_t)nRows, (int64_t)nRows,
                                        workspace->dDenseA, CUDA_R_64F, CUSPARSE_ORDER_COL));

    checkCudaErrors(cusparseSparseToDense_bufferSize(cusparseHandle, workspace->sparseMatrix, workspace->denseMatrix,
                                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &workspace->sparseToDenseBufferSize));
    checkCudaErrors(cudaMalloc((void **)&workspace->dSparseToDenseBuffer, workspace->sparseToDenseBufferSize));

    checkCudaErrors(cusolverDnDgetrf_bufferSize(cusolverDnHandle,
                                                nRows, nRows,
                                                workspace->dDenseA, nRows,
                                                &workspace->luWorkSize));
    checkCudaErrors(cudaMalloc((void **)&workspace->dLuWork, sizeof(double) * (size_t)workspace->luWorkSize));
    checkCudaErrors(cudaMalloc((void **)&workspace->dLuPivot, sizeof(int) * (size_t)nRows));
    checkCudaErrors(cudaMalloc((void **)&workspace->dLuInfo, sizeof(int)));
}

void releaseDenseLinearSolveWorkspace(DenseLinearSolveWorkspace *workspace)
{
    if (workspace->sparseMatrix)
    {
        checkCudaErrors(cusparseDestroySpMat(workspace->sparseMatrix));
        workspace->sparseMatrix = NULL;
    }
    if (workspace->denseMatrix)
    {
        checkCudaErrors(cusparseDestroyDnMat(workspace->denseMatrix));
        workspace->denseMatrix = NULL;
    }
    if (workspace->dDenseA)
    {
        checkCudaErrors(cudaFree(workspace->dDenseA));
        workspace->dDenseA = NULL;
    }
    if (workspace->dLuWork)
    {
        checkCudaErrors(cudaFree(workspace->dLuWork));
        workspace->dLuWork = NULL;
    }
    if (workspace->dLuPivot)
    {
        checkCudaErrors(cudaFree(workspace->dLuPivot));
        workspace->dLuPivot = NULL;
    }
    if (workspace->dLuInfo)
    {
        checkCudaErrors(cudaFree(workspace->dLuInfo));
        workspace->dLuInfo = NULL;
    }
    if (workspace->dSparseToDenseBuffer)
    {
        checkCudaErrors(cudaFree(workspace->dSparseToDenseBuffer));
        workspace->dSparseToDenseBuffer = NULL;
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
    checkCudaErrors(cudaMemcpyAsync(dSolution, dRhs, sizeof(double) * (size_t)workspace->nRows,
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
    checkCudaErrors(cudaMemcpy(&info, workspace->dLuInfo, sizeof(int), cudaMemcpyDeviceToHost));
    return info == 0;
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

bool solveDenseLinearSystemFactored(DenseLinearSolveWorkspace *workspace,
                                     const double *dRhs,
                                     double *dSolution,
                                     cusolverDnHandle_t cusolverDnHandle,
                                     cudaStream_t cudaStream)
{
    int info = 0;
    checkCudaErrors(cudaMemcpyAsync(dSolution, dRhs, sizeof(double) * (size_t)workspace->nRows,
                                    cudaMemcpyDeviceToDevice, cudaStream));
    checkCudaErrors(cusolverDnDgetrs(cusolverDnHandle, CUBLAS_OP_N,
                                     workspace->nRows, 1,
                                     workspace->dDenseA, workspace->nRows,
                                     workspace->dLuPivot, dSolution, workspace->nRows,
                                     workspace->dLuInfo));
    checkCudaErrors(cudaMemcpy(&info, workspace->dLuInfo, sizeof(int), cudaMemcpyDeviceToHost));
    return info == 0;
}

// CUDA 12+ cuSPARSE: legacy enums removed; use generic API enums
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
#ifndef CUSPARSE_CSRMV_ALG2
#define CUSPARSE_CSRMV_ALG2 CUSPARSE_SPMV_CSR_ALG2
#endif
#ifndef CUSPARSE_CSR2CSC_ALG2
#define CUSPARSE_CSR2CSC_ALG2 CUSPARSE_CSR2CSC_ALG1
#endif
#ifndef CUSPARSE_CSRMM_ALG1
#define CUSPARSE_CSRMM_ALG1 CUSPARSE_SPMM_ALG_DEFAULT
#endif
#endif

SyphaStatus solver_sparse_mehrotra(SyphaNodeSparse &node)
{
    SolverExecutionConfig config;
    config.maxIterations = node.env->mehrotraMaxIter;
    config.gapStagnation.enabled = false;
    config.bnbNodeOrdinal = 0;
    config.denseSelectionLogEveryNodes = 1;

    SolverExecutionResult result;
    SyphaStatus status = solver_sparse_mehrotra_run(node, config, &result);
    if (status != CODE_SUCCESFULL)
    {
        return status;
    }
    return result.status;
}

SyphaStatus solver_sparse_mehrotra_run(SyphaNodeSparse &node, const SolverExecutionConfig &config, SolverExecutionResult *result)
{
    const int reorder = node.env->mehrotraReorder;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    size_t currBufferSize = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = NULL;
    double *d_bufferS = NULL;
    double *d_buffer = NULL;
    double *d_tmp_prim = NULL;
    double *d_tmp_dual = NULL;
    double *d_blockmin_prim = NULL;
    double *d_blockmin_dual = NULL;
    int nBlocksAlpha = 0;
    char message[1024];

    cusparseMatDescr_t A_descr;
    DenseLinearSolveWorkspace denseLinearWorkspace;


    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.hX = (double *)malloc(sizeof(double) * node.ncols);
    node.hY = (double *)malloc(sizeof(double) * node.nrows);
    node.hS = (double *)malloc(sizeof(double) * node.ncols);

    node.timeStartSolStart = node.env->timer();
    solver_sparse_mehrotra_init_gsl(node);
    node.timeStartSolEnd = node.env->timer();

    ///////////////////             SET BIG MATRIX ON HOST
    //
    // On each step we solve this linear system twice:
    //
    //      O | A' | I    x    -rc
    //      --|----|---   -    ---
    //      A | O  | O  * y  = -rb
    //      --|----|---   -    ---
    //      S | O  | X    s    -rxs
    //
    // Where A is the model matrix (standard form), I is the n*n identity
    // matrix, S is the n*n s diagonal matrix, X is the n*n diagonal matrix.
    // Total number of non-zero elements is A.nnz * 2 + n * 3

    node.timePreSolStart = node.env->timer();

    int A_nrows = node.ncols * 2 + node.nrows;
    int A_ncols = A_nrows;
    int A_nnz = node.nnz * 2 + node.ncols * 3;
    const std::uint64_t denseKktBytes = (std::uint64_t)A_nrows * (std::uint64_t)A_ncols * (std::uint64_t)sizeof(double);
    const std::uint64_t sparseKktBytes =
        (std::uint64_t)A_nnz * ((std::uint64_t)sizeof(double) + (std::uint64_t)sizeof(int)) +
        (std::uint64_t)(A_nrows + 1) * (std::uint64_t)sizeof(int);
    bool useDenseLinearSolve = false;

    int *h_csrAInds = NULL;
    int *h_csrAOffs = NULL;
    double *h_csrAVals = NULL;

    int *d_csrAInds = NULL;
    int *d_csrAOffs = NULL;
    double *d_csrAVals = NULL;

    double *d_rhs = NULL;
    double *d_sol = NULL;
    double *d_prevSol = NULL;

    h_csrAInds = (int *)calloc(sizeof(int), A_nnz);
    h_csrAOffs = (int *)calloc(sizeof(int), (A_nrows + 1));
    h_csrAVals = (double *)calloc(sizeof(double), A_nnz);

    sprintf(message, "Matrix: %d x %d, %d non-zeros", A_nrows, A_ncols, A_nnz);
    node.env->logger(message, "INFO", 17);

    // Instantiate the first group of n rows: O | A' | I
    bool found = false;
    int off = 0, rowCnt = 0;

    h_csrAOffs[0] = 0;
    for (j = 0; j < node.ncols; ++j)
    {
        rowCnt = 0;
        for (i = 0; i < node.nrows; ++i)
        {
            found = false;
            for (k = node.hCsrMatOffs->data()[i]; k < node.hCsrMatOffs->data()[i + 1]; ++k)
            {
                if (node.hCsrMatInds->data()[k] == j)
                {
                    found = true;
                    break;
                }
            }

            if (found)
            {
                h_csrAInds[off] = node.ncols + i;
                h_csrAVals[off] = node.hCsrMatVals->data()[k];
                ++rowCnt;
                ++off;
            }
        }

        // append the I matrix element for the current row
        h_csrAInds[off] = node.ncols + node.nrows + j;
        h_csrAVals[off] = 1.0;
        ++rowCnt;
        ++off;

        h_csrAOffs[j + 1] = h_csrAOffs[j] + rowCnt;
    }

    // Instantiate the second group of m rows: A | O | O
    for (i = 0; i < node.nrows; ++i)
    {
        h_csrAOffs[node.ncols + i + 1] = h_csrAOffs[node.ncols + i] + (node.hCsrMatOffs->data()[i + 1] - node.hCsrMatOffs->data()[i]);
    }
    memcpy(&h_csrAInds[off], node.hCsrMatInds->data(), sizeof(int) * node.nnz);
    memcpy(&h_csrAVals[off], node.hCsrMatVals->data(), sizeof(double) * node.nnz);
    off += node.nnz;

    // Instantiate the third group of n rows: S | O | X
    for (j = 0; j < node.ncols; ++j)
    {
        // s
        h_csrAInds[off] = j;
        h_csrAVals[off] = node.hS[j];
        ++off;

        // x
        h_csrAInds[off] = node.ncols + node.nrows + j;
        h_csrAVals[off] = node.hX[j];
        ++off;

        h_csrAOffs[node.ncols + node.nrows + j + 1] = h_csrAOffs[node.ncols + node.nrows + j] + 2;
    }

    checkCudaErrors(cudaMalloc((void **)&d_csrAInds, sizeof(int) * A_nnz));
    checkCudaErrors(cudaMalloc((void **)&d_csrAOffs, sizeof(int) * (A_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrAVals, sizeof(double) * A_nnz));

    checkCudaErrors(cudaMemcpy(d_csrAInds, h_csrAInds, sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAOffs, h_csrAOffs, sizeof(int) * (A_nrows + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAVals, h_csrAVals, sizeof(double) * A_nnz, cudaMemcpyHostToDevice));

    checkCudaErrors(cusparseCreateMatDescr(&A_descr));
    checkCudaErrors(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));

    size_t gpuMemFree = 0;
    size_t gpuMemTotal = 0;
    const bool canTrackGpuMem = (cudaMemGetInfo(&gpuMemFree, &gpuMemTotal) == cudaSuccess);
    const size_t gpuMemFreeBeforeKktSetup = gpuMemFree;
    size_t gpuMemFreeAfterKktSetup = gpuMemFree;
    size_t gpuMemMinDuringLinearSolve = gpuMemFree;
    int gpuMemSampleCount = 0;

    if ((node.env->denseGpuMemoryFractionThreshold > 0.0) && canTrackGpuMem)
    {
        const double thresholdBytes = node.env->denseGpuMemoryFractionThreshold * (double)gpuMemTotal;
        useDenseLinearSolve = (double)denseKktBytes < thresholdBytes;
    }

    const int logEveryNodes = config.denseSelectionLogEveryNodes <= 0 ? 1 : config.denseSelectionLogEveryNodes;
    const bool shouldLogDenseSelection =
        (config.bnbNodeOrdinal <= 0) ||
        (logEveryNodes <= 1) ||
        ((config.bnbNodeOrdinal % logEveryNodes) == 0);

    if (useDenseLinearSolve)
    {
        initializeDenseLinearSolveWorkspace(&denseLinearWorkspace, A_nrows, A_nnz,
                                            d_csrAOffs, d_csrAInds, d_csrAVals,
                                            node.cusparseHandle, node.cusolverDnHandle);
        if (shouldLogDenseSelection)
        {
            sprintf(message,
                    "Using dense linear solver (dense KKT %.2f MB, sparse KKT %.2f MB, threshold %.2f MB)",
                    (double)denseKktBytes / (1024.0 * 1024.0),
                    (double)sparseKktBytes / (1024.0 * 1024.0),
                    (node.env->denseGpuMemoryFractionThreshold * (double)gpuMemTotal) / (1024.0 * 1024.0));
            node.env->logger(message, "INFO", 5);
        }
    }
    else
    {
        if (shouldLogDenseSelection)
        {
            sprintf(message,
                    "Using sparse linear solver (sparse KKT %.2f MB, dense KKT %.2f MB)",
                    (double)sparseKktBytes / (1024.0 * 1024.0),
                    (double)denseKktBytes / (1024.0 * 1024.0));
            node.env->logger(message, "INFO", 10);
        }
    }

    auto sampleGpuMemory = [&]() {
        if (!canTrackGpuMem)
        {
            return;
        }
        size_t freeNow = 0;
        size_t totalNow = 0;
        if (cudaMemGetInfo(&freeNow, &totalNow) == cudaSuccess)
        {
            if (freeNow < gpuMemMinDuringLinearSolve)
            {
                gpuMemMinDuringLinearSolve = freeNow;
            }
            ++gpuMemSampleCount;
        }
    };

    sampleGpuMemory();
    gpuMemFreeAfterKktSetup = gpuMemMinDuringLinearSolve;

    free(h_csrAInds);
    free(h_csrAOffs);
    free(h_csrAVals);

    ///////////////////             INITIALISE RHS

    checkCudaErrors(cudaMalloc((void **)&d_rhs, sizeof(double) * A_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_sol, sizeof(double) * A_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_prevSol, sizeof(double) * A_nrows));

    // put x, y, s on device sol as [x, y, s]
    double *d_x = d_prevSol;
    double *d_y = &d_prevSol[node.ncols];
    double *d_s = &d_prevSol[node.ncols + node.nrows];

    double *d_deltaX = d_sol;
    double *d_deltaY = &d_sol[node.ncols];
    double *d_deltaS = &d_sol[node.ncols + node.nrows];

    checkCudaErrors(cudaMemcpy(d_x, node.hX, sizeof(double) * node.ncols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, node.hY, sizeof(double) * node.nrows, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_s, node.hS, sizeof(double) * node.ncols, cudaMemcpyHostToDevice));

    // put OBJ and S on device rhs
    double *d_resC = d_rhs;
    double *d_resB = &d_rhs[node.ncols];
    double *d_resXS = &d_rhs[node.ncols + node.nrows];

    checkCudaErrors(cudaMemcpy(d_resC, node.dObjDns, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_resB, node.dRhsDns, sizeof(double) * node.nrows, cudaMemcpyDeviceToDevice));

    // Residuals
    // resB, resC equation 14.7, page 395(414)Numerical Optimization
    // resC = -mat' * y + (obj - s)
    // resB = -mat  * x + rhs

    cusparseDnVecDescr_t vecX, vecY, vecResC, vecResB;

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols, d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows, d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResC, (int64_t)node.ncols, d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, (int64_t)node.nrows, d_resB, CUDA_R_64F));

    alpha = -1.0;
    checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                &alpha, d_s, 1, d_resC, 1));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, vecY,
                                            &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    // buffer size for other needs
    currBufferSize = (size_t)(sizeof(double) * node.ncols * 2);
    currBufferSize = currBufferSize > bufferSize ? currBufferSize : bufferSize;
    checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));

    checkCudaErrors(cusparseSpMV(node.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, vecY,
                                 &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, node.matDescr, vecX,
                                            &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, node.matDescr, vecX,
                                 &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 (size_t *)d_buffer));

    ///////////////////             CALCULATE MU
    // duality measure, defined at page 395(414) Numerical Optimization
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
    mu /= node.ncols;

    node.timePreSolEnd = node.env->timer();

    // Alpha-max reduction buffers (GPU step-length)
    nBlocksAlpha = (node.ncols + 255) / 256;
    checkCudaErrors(cudaMalloc((void **)&d_tmp_prim, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_tmp_dual, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_blockmin_prim, sizeof(double) * nBlocksAlpha));
    checkCudaErrors(cudaMalloc((void **)&d_blockmin_dual, sizeof(double) * nBlocksAlpha));

    ///////////////////             MAIN LOOP

    node.env->logger("Mehrotra procedure started", "INFO", 17);
    node.timeSolverStart = node.env->timer();
    const int maxIterations = config.maxIterations > 0 ? config.maxIterations : node.env->mehrotraMaxIter;
    const bool gapStagnationEnabled = config.gapStagnation.enabled && config.gapStagnation.windowIterations > 0 && config.gapStagnation.minImprovementPct >= 0.0;
    const double minImprovementRatio = config.gapStagnation.minImprovementPct / 100.0;
    int noGapImprovementIterations = 0;
    double bestRelativeGap = std::numeric_limits<double>::infinity();
    SolverTerminationReason terminationReason = SOLVER_TERM_MAX_ITER;
    bool infeasibleOrNumerical = false;

    while ((iterations < maxIterations) && (mu > node.env->mehrotraMuTol))
    {

        // x, s multiplication: -x.*s on device (was host + 3 full-vector PCIe transfers)
        elem_min_mult_dev(d_x, d_s, d_resXS, node.ncols);

        // Dense path: factorize once, then reuse for both affine and corrector solves.
        // Sparse path: cusolverSpDcsrlsvqr does not expose separate factor/solve, so
        // we call the monolithic solver for each RHS.
        if (useDenseLinearSolve)
        {
            sampleGpuMemory();
            if (!factorizeDenseLinearSystem(&denseLinearWorkspace,
                                            node.cusparseHandle, node.cusolverDnHandle))
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                break;
            }
            sampleGpuMemory();
            if (!solveDenseLinearSystemFactored(&denseLinearWorkspace, d_rhs, d_sol,
                                                 node.cusolverDnHandle, node.cudaStream))
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                break;
            }
        }
        else
        {
            sampleGpuMemory();
            checkCudaErrors(cusolverSpDcsrlsvqr(node.cusolverSpHandle,
                                                A_nrows, A_nnz, A_descr,
                                                d_csrAVals, d_csrAOffs, d_csrAInds,
                                                d_rhs,
                                                node.env->mehrotraCholTol, reorder,
                                                d_sol, &singularity));
            sampleGpuMemory();
            if (singularity >= 0)
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                break;
            }
        }

        // affine step length: alphaMaxPrim = min(-x/dx for dx<0), alphaMaxDual = min(-s/ds for ds<0) on device
        alpha_max_dev(d_x, d_deltaX, d_s, d_deltaS, node.ncols,
                      d_tmp_prim, d_tmp_dual, d_blockmin_prim, d_blockmin_dual,
                      &alphaMaxPrim, &alphaMaxDual);

        alphaPrim = gsl_min(1.0, alphaMaxPrim);
        alphaDual = gsl_min(1.0, alphaMaxDual);

        // mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s + alpha_aff_d * delta_s_aff) / float(n)
        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system
        // the dimension of the buffer is guaranteed to be >= 2 * ncols
        d_bufferX = d_buffer;
        d_bufferS = &d_buffer[node.ncols];
        checkCudaErrors(cudaMemcpyAsync(d_bufferX, d_x, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
        checkCudaErrors(cudaMemcpyAsync(d_bufferS, d_s, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaPrim, d_deltaX, 1, d_bufferX, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaDual, d_deltaS, 1, d_bufferS, 1));

        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_bufferX, 1, d_bufferS, 1, &muAff));
        muAff /= node.ncols;

        // corrector step or centering parameter
        sigma = gsl_pow_3(muAff / mu);

        // d_bufferX = -deltaX.*deltaS + sigma*mu (on device)
        corrector_rhs_dev(d_deltaX, d_deltaS, sigma, mu, d_bufferX, node.ncols);

        alpha = 1.0;
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alpha, d_bufferX, 1, d_resXS, 1));

        // Solve corrector step (dense: triangular solve reusing LU factors;
        // sparse: full QR solve — cusolverSpDcsrlsvqr has no separate factor/solve)
        if (useDenseLinearSolve)
        {
            sampleGpuMemory();
            if (!solveDenseLinearSystemFactored(&denseLinearWorkspace, d_rhs, d_sol,
                                                 node.cusolverDnHandle, node.cudaStream))
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                break;
            }
        }
        else
        {
            sampleGpuMemory();
            checkCudaErrors(cusolverSpDcsrlsvqr(node.cusolverSpHandle,
                                                A_nrows, A_nnz, A_descr,
                                                d_csrAVals, d_csrAOffs, d_csrAInds,
                                                d_rhs,
                                                node.env->mehrotraCholTol, reorder,
                                                d_sol, &singularity));
            if (singularity >= 0)
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                break;
            }
        }
        sampleGpuMemory();

        // corrector step length: alpha-max on device
        alpha_max_dev(d_x, d_deltaX, d_s, d_deltaS, node.ncols,
                      d_tmp_prim, d_tmp_dual, d_blockmin_prim, d_blockmin_dual,
                      &alphaMaxPrim, &alphaMaxDual);

        alphaPrim = gsl_min(1.0, node.env->mehrotraEta * alphaMaxPrim);
        alphaDual = gsl_min(1.0, node.env->mehrotraEta * alphaMaxDual);

        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaPrim, d_deltaX, 1, d_x, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.nrows,
                                    &alphaDual, d_deltaY, 1, d_y, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaDual, d_deltaS, 1, d_s, 1));

        ///////////////             UPDATE

        alpha = -(alphaDual - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.ncols,
                                    &alpha, d_resC, 1));

        alpha = -(alphaPrim - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.nrows,
                                    &alpha, d_resB, 1));

        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
        mu /= node.ncols;
        if (!std::isfinite(mu) || (mu < 0.0))
        {
            infeasibleOrNumerical = true;
            terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
            break;
        }

        // update x and s on matrix
        off = node.nnz * 2 + node.ncols;
        checkCudaErrors(cublasDcopy(node.cublasHandle, node.ncols, d_s, 1, &d_csrAVals[off], 2));
        checkCudaErrors(cublasDcopy(node.cublasHandle, node.ncols, d_x, 1, &d_csrAVals[off + 1], 2));

        // Update primal-dual gap monitor
        double currPrimalObj = 0.0;
        double currDualObj = 0.0;
        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncolsOriginal,
                                   d_x, 1, node.dObjDns, 1, &currPrimalObj));
        checkCudaErrors(cublasDdot(node.cublasHandle, node.nrows,
                                   d_y, 1, node.dRhsDns, 1, &currDualObj));
        const double denom = std::max(1.0, fabs(currPrimalObj));
        const double relativeGap = fabs(currPrimalObj - currDualObj) / denom;
        if (!std::isfinite(currPrimalObj) || !std::isfinite(currDualObj) || !std::isfinite(relativeGap))
        {
            infeasibleOrNumerical = true;
            terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
            break;
        }

        if (relativeGap < bestRelativeGap * (1.0 - minImprovementRatio))
        {
            bestRelativeGap = relativeGap;
            noGapImprovementIterations = 0;
        }
        else if (gapStagnationEnabled)
        {
            ++noGapImprovementIterations;
            if (noGapImprovementIterations >= config.gapStagnation.windowIterations)
            {
                terminationReason = SOLVER_TERM_GAP_STALLED;
                ++iterations;
                break;
            }
        }

        ++iterations;
    }

    node.iterations = iterations;
    if (!infeasibleOrNumerical && (terminationReason != SOLVER_TERM_GAP_STALLED))
    {
        terminationReason = (mu <= node.env->mehrotraMuTol) ? SOLVER_TERM_CONVERGED : SOLVER_TERM_MAX_ITER;
    }

    // Compute objective using only original variables (not slack variables)
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncolsOriginal,
                               d_x, 1, node.dObjDns, 1, &node.objvalPrim));

    checkCudaErrors(cublasDdot(node.cublasHandle, node.nrows,
                               d_y, 1, node.dRhsDns, 1, &node.objvalDual));
    const double finalGapDenom = std::max(1.0, fabs(node.objvalPrim));
    const double finalRelativeGap = fabs(node.objvalPrim - node.objvalDual) / finalGapDenom;
    const double finalBoundViolation = node.objvalDual - node.objvalPrim;
    if (!infeasibleOrNumerical &&
        (terminationReason != SOLVER_TERM_CONVERGED) &&
        std::isfinite(finalBoundViolation) &&
        (finalBoundViolation > (1e6 * std::max(1.0, fabs(node.objvalPrim)))))
    {
        infeasibleOrNumerical = true;
        terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
    }
    node.mipGap = std::numeric_limits<double>::infinity();

    if (infeasibleOrNumerical)
    {
        node.env->logger("LP relaxation flagged as infeasible or numerically unstable", "INFO", 5);
    }

    if (canTrackGpuMem)
    {
        const double toMb = 1024.0 * 1024.0;
        const double freeBeforeMb = (double)gpuMemFreeBeforeKktSetup / toMb;
        const double freeAfterSetupMb = (double)gpuMemFreeAfterKktSetup / toMb;
        const double minDuringSolveMb = (double)gpuMemMinDuringLinearSolve / toMb;
        const double setupUsedMb = (double)(gpuMemFreeBeforeKktSetup >= gpuMemFreeAfterKktSetup ? (gpuMemFreeBeforeKktSetup - gpuMemFreeAfterKktSetup) : 0) / toMb;
        const double extraPeakMb = (double)(gpuMemFreeAfterKktSetup >= gpuMemMinDuringLinearSolve ? (gpuMemFreeAfterKktSetup - gpuMemMinDuringLinearSolve) : 0) / toMb;
        sprintf(message,
                "GPU memory telemetry (%s): free before KKT %.2f MB, free after setup %.2f MB, min free during solves %.2f MB, setup used %.2f MB, extra solve peak %.2f MB, samples %d",
                useDenseLinearSolve ? "dense" : "sparse",
                freeBeforeMb, freeAfterSetupMb, minDuringSolveMb, setupUsedMb, extraPeakMb, gpuMemSampleCount);
        node.env->logger(message, "INFO", 8);
    }

    node.env->logger("Mehrotra procedure finished", "INFO", 10);
    node.timeSolverEnd = node.env->timer();

    if (result != NULL)
    {
        result->status = infeasibleOrNumerical ? CODE_GENERIC_ERROR : CODE_SUCCESFULL;
        result->terminationReason = terminationReason;
        result->iterations = iterations;
        result->primalObj = node.objvalPrim;
        result->dualObj = node.objvalDual;
        result->relativeGap = finalRelativeGap;
        result->primalSolution.resize((size_t)node.ncols, 0.0);
        result->dualSolution.resize((size_t)node.nrows, 0.0);
        checkCudaErrors(cudaMemcpy(result->primalSolution.data(), d_x,
                                   sizeof(double) * node.ncols, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(result->dualSolution.data(), d_y,
                                   sizeof(double) * node.nrows, cudaMemcpyDeviceToHost));
    }

    ///////////////////             RELEASE RESOURCES

    checkCudaErrors(cusparseDestroyMatDescr(A_descr));

    checkCudaErrors(cudaFree(d_csrAInds));
    checkCudaErrors(cudaFree(d_csrAOffs));
    checkCudaErrors(cudaFree(d_csrAVals));

    checkCudaErrors(cudaFree(d_rhs));
    checkCudaErrors(cudaFree(d_sol));
    checkCudaErrors(cudaFree(d_prevSol));

    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecResC));
    checkCudaErrors(cusparseDestroyDnVec(vecResB));

    if (node.matTransDescr)
    {
        checkCudaErrors(cusparseDestroySpMat(node.matTransDescr));
        node.matTransDescr = NULL;
    }

    if (d_tmp_prim)
        checkCudaErrors(cudaFree(d_tmp_prim));
    if (d_tmp_dual)
        checkCudaErrors(cudaFree(d_tmp_dual));
    if (d_blockmin_prim)
        checkCudaErrors(cudaFree(d_blockmin_prim));
    if (d_blockmin_dual)
        checkCudaErrors(cudaFree(d_blockmin_dual));
    if (d_buffer)
        checkCudaErrors(cudaFree(d_buffer));

    if (denseLinearWorkspace.isEnabled)
    {
        releaseDenseLinearSolveWorkspace(&denseLinearWorkspace);
    }

    return infeasibleOrNumerical ? CODE_GENERIC_ERROR : CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    size_t currBufferSize = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = NULL;
    double *d_bufferS = NULL;
    double *d_buffer = NULL;

    double *d_resC = NULL, *d_resB = NULL, *d_resXS = NULL;
    double *d_x = NULL, *d_y = NULL, *d_s = NULL;
    double *d_delX = NULL, *d_delY = NULL, *d_delS = NULL;
    char message[1024];

    cusparseSpMatDescr_t spMatTransDescr;
    cusparseDnVecDescr_t vecX, vecY, vecResC, vecResB;

    ///////////////////             GET TRANSPOSED MATRIX

    int *d_csrMatTransOffs = NULL, *d_csrMatTransInds = NULL;
    double *d_csrMatTransVals = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_csrMatTransOffs, sizeof(int) * (node.ncols + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrMatTransInds, sizeof(int) * node.nnz));
    checkCudaErrors(cudaMalloc((void **)&d_csrMatTransVals, sizeof(double) * node.nnz));

    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                                  node.dCsrMatVals, node.dCsrMatOffs, node.dCsrMatInds,
                                                  d_csrMatTransVals, d_csrMatTransOffs, d_csrMatTransInds,
                                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                                  &bufferSize));
    // buffer size for other needs
    currBufferSize = (size_t)(sizeof(double) * node.ncols * 2);
    currBufferSize = currBufferSize > bufferSize ? currBufferSize : bufferSize;
    checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));

    checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                       node.dCsrMatVals, node.dCsrMatOffs, node.dCsrMatInds,
                                       d_csrMatTransVals, d_csrMatTransOffs, d_csrMatTransInds,
                                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                       CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                       d_buffer));

    checkCudaErrors(cusparseCreateCsr(&spMatTransDescr, node.ncols, node.nrows, node.nnz,
                                      d_csrMatTransOffs, d_csrMatTransInds, d_csrMatTransVals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.hX = (double *)malloc(sizeof(double) * node.ncols);
    node.hY = (double *)malloc(sizeof(double) * node.nrows);
    node.hS = (double *)malloc(sizeof(double) * node.ncols);

    node.timeStartSolStart = node.env->timer();
    solver_sparse_mehrotra_init_gsl(node);
    node.timeStartSolEnd = node.env->timer();

    ///////////////////             INITIALISE RHS

    node.timePreSolStart = node.env->timer();

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_s, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_x, node.hX, sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_y, node.hY, sizeof(double) * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_s, node.hS, sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));

    // put OBJ and S on device rhs
    checkCudaErrors(cudaMalloc((void **)&d_resC, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_resB, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_resXS, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_resC, node.dObjDns, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_resB, node.dRhsDns, sizeof(double) * node.nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

    // Residuals
    // resB, resC equation 14.7, page 395(414)Numerical Optimization
    // resC = -mat' * y + (obj - s)
    // resB = -mat  * x + rhs

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols, d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows, d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResC, (int64_t)node.ncols, d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, (int64_t)node.nrows, d_resB, CUDA_R_64F));

    alpha = -1.0;
    checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                &alpha, d_s, 1, d_resC, 1));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, spMatTransDescr, vecY,
                                            &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    // buffer size for other needs
    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, spMatTransDescr, vecY,
                                 &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, node.matDescr, vecX,
                                            &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, node.matDescr, vecX,
                                 &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 (size_t *)d_buffer));

    ///////////////////             CALCULATE MU
    // duality measure, defined at page 395(414) Numerical Optimization
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
    mu /= node.ncols;

    node.timePreSolEnd = node.env->timer();

    ///////////////////             MAIN LOOP

    node.env->logger("Mehrotra procedure started", "INFO", 17);
    node.timeSolverStart = node.env->timer();

    iterations = 0;

    while ((iterations < node.env->mehrotraMaxIter) && (mu > node.env->mehrotraMuTol))
    {

        ++iterations;
    }

    node.timeSolverEnd = node.env->timer();

    ///////////////////             RELEASE RESOURCES

    free(node.hX);
    free(node.hY);
    free(node.hS);

    cusparseDestroySpMat(spMatTransDescr);

    checkCudaErrors(cudaFree(d_csrMatTransOffs));
    checkCudaErrors(cudaFree(d_csrMatTransInds));
    checkCudaErrors(cudaFree(d_csrMatTransVals));

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_s));

    checkCudaErrors(cudaFree(d_resC));
    checkCudaErrors(cudaFree(d_resB));
    checkCudaErrors(cudaFree(d_resXS));

    return CODE_SUCCESFULL;
}

#if !(defined(CUDART_VERSION) && CUDART_VERSION >= 12000)
SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int64_t AAT_nrows = node.nrows, AAT_ncols = node.nrows, AAT_nnz = 0;
    double alpha = 1.0;
    double beta = 0.0;

    int *AAT_inds = NULL, *AAT_offs = NULL;
    double *AAT_vals = NULL;

    void *d_buffer1 = NULL, *d_buffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    cusparseSpMatDescr_t AAT_descr;
    cusparseMatDescr_t AAT_descrGen, matTransDescrGen;
    cusparseSpGEMMDescr_t spgemmDescr;

    checkCudaErrors(cusparseCreateMatDescr(&AAT_descrGen));
    checkCudaErrors(cusparseSetMatType(AAT_descrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(AAT_descrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateMatDescr(&matTransDescrGen));
    checkCudaErrors(cusparseSetMatType(matTransDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matTransDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    ///////////////////             COMPUTE STARTING COORDINATES X AND S

    // AAT matrix for geMM
    checkCudaErrors(cusparseCreateCsr(&AAT_descr, AAT_nrows, AAT_ncols, AAT_nnz,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // SpGEMM Computation
    checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

    // ask bufferSize1 bytes for external memory
    checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, node.matTransDescr,
                                                  &beta, AAT_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, bufferSize1));

    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, node.matTransDescr,
                                                  &beta, AAT_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, d_buffer1));

    // ask bufferSize2 bytes for external memory
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, node.matTransDescr,
                                           &beta, AAT_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_buffer2, bufferSize2));

    // compute the intermediate product of A * B
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, node.matTransDescr,
                                           &beta, AAT_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

    // get matrix C non-zero entries C_num_nnz1
    // cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);
    cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);

    // allocate matrix AAT
    checkCudaErrors(cudaMalloc((void **)&AAT_offs, sizeof(int) * (AAT_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&AAT_inds, sizeof(int) * AAT_nnz));
    checkCudaErrors(cudaMalloc((void **)&AAT_vals, sizeof(double) * AAT_nnz));

    // update AAT with the new pointers
    checkCudaErrors(cusparseCsrSetPointers(AAT_descr, AAT_offs, AAT_inds, AAT_vals));

    // copy the final products to the matrix AAT
    checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, node.matDescr, node.matTransDescr,
                                        &beta, AAT_descr,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

    checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));

    ///////////////////             COMPUTE s = - mat' * y + obj
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.dS, node.dObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseCsrmvEx_bufferSize(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               node.ncols, node.nrows, node.nnz,
                                               &alpha, CUDA_R_64F,
                                               matTransDescrGen,
                                               node.dCsrMatTransVals, CUDA_R_64F,
                                               node.dCsrMatTransOffs,
                                               node.dCsrMatTransInds,
                                               node.dY, CUDA_R_64F,
                                               &beta, CUDA_R_64F,
                                               node.dS, CUDA_R_64F, CUDA_R_64F,
                                               &bufferSize1));

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, bufferSize1));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseCsrmvEx(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    node.ncols, node.nrows, node.nnz,
                                    &alpha, CUDA_R_64F,
                                    matTransDescrGen,
                                    node.dCsrMatTransVals, CUDA_R_64F,
                                    node.dCsrMatTransOffs,
                                    node.dCsrMatTransInds,
                                    node.dY, CUDA_R_64F,
                                    &beta, CUDA_R_64F,
                                    node.dS, CUDA_R_64F, CUDA_R_64F,
                                    d_buffer1));

    checkCudaErrors(cusparseDestroyMatDescr(AAT_descrGen));
    checkCudaErrors(cusparseDestroyMatDescr(matTransDescrGen));
    checkCudaErrors(cusparseDestroySpMat(AAT_descr));

    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));

    checkCudaErrors(cudaFree(AAT_inds));
    checkCudaErrors(cudaFree(AAT_offs));
    checkCudaErrors(cudaFree(AAT_vals));

    return CODE_SUCCESFULL;
}
#endif // !(CUDA 12+)

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
// CUDA 12+: legacy csrmvEx / CUSPARSE_ALG_MERGE_PATH are unavailable.
// Reuse the GSL-based initialisation instead of the legacy cuSPARSE path.
SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node)
{
    return solver_sparse_mehrotra_init_gsl(node);
}
#endif

SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;
    int info = 0;
    int i = 0;
    int I_matBytes = node.nrows * node.nrows * sizeof(double);

    double alpha = 1.0;
    double beta = 0.0;

    int *d_ipiv = NULL;
    double *d_AAT = NULL;
    double *d_matDn = NULL;
    double *h_I = NULL;

    void *d_buffer = NULL;
    size_t currBufferSize = 0;
    size_t bufferSize = 0;

    cusolverDnParams_t cusolverDnParams;
    cusparseDnVecDescr_t vecX, vecY, vecS;
    cusparseDnMatDescr_t AAT_descr, matDnDescr;
    cusparseMatDescr_t matDescrGen;

    node.env->logger("Mehrotra starting point computation", "INFO", 13);
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    checkCudaErrors(cusparseCreateMatDescr(&matDescrGen));
    checkCudaErrors(cusparseSetMatType(matDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols,
                                        node.dX, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows,
                                        node.dY, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecS, (int64_t)node.ncols,
                                        node.dS, CUDA_R_64F));

    checkCudaErrors(cudaMalloc((void **)&d_AAT, sizeof(double) * node.nrows * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_matDn, sizeof(double) * node.nrows * node.ncols));

    checkCudaErrors(cusparseCreateDnMat(&AAT_descr, (int64_t)node.nrows, (int64_t)node.nrows,
                                        (int64_t)node.nrows, d_AAT, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    checkCudaErrors(cusparseCreateDnMat(&matDnDescr, (int64_t)node.nrows, (int64_t)node.ncols,
                                        (int64_t)node.nrows, d_matDn, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    ///////////////////             STORE MATRIX IN DENSE FORMAT
    node.env->logger("Init: storing matrix (dense)", "INFO", 20);
    size_t szSparseToDense = 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    checkCudaErrors(cusparseSparseToDense_bufferSize(node.cusparseHandle, node.matDescr, matDnDescr,
                                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &szSparseToDense));
#endif
    checkCudaErrors(cusparseSpMM_bufferSize(node.cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, matDnDescr,
                                            &beta, AAT_descr,
                                            CUDA_R_64F,
                                            CUSPARSE_CSRMM_ALG1,
                                            &bufferSize));

    // allocate memory for computation
    currBufferSize = bufferSize > I_matBytes ? bufferSize : I_matBytes;
    if (szSparseToDense > currBufferSize)
        currBufferSize = szSparseToDense;
    checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    checkCudaErrors(cusparseSparseToDense(node.cusparseHandle, node.matDescr, matDnDescr,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buffer));
#else
    checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, node.nrows, node.ncols,
                                       matDescrGen,
                                       node.dCsrMatVals, node.dCsrMatOffs, node.dCsrMatInds,
                                       d_matDn, node.nrows));
#endif

    ///////////////////             COMPUTE AAT INVERSE MATRIX

    // GEMM Computation: MATRIX * MATRIX'

    checkCudaErrors(cusparseSpMM(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, matDnDescr,
                                 &beta, AAT_descr,
                                 CUDA_R_64F,
                                 CUSPARSE_CSRMM_ALG1,
                                 d_buffer));

    ///////////////////             MATRIX INVERSION

    // See https://stackoverflow.com/questions/50892906/what-is-the-most-efficient-way-to-compute-the-inverse-of-a-general-matrix-using
    checkCudaErrors(cusolverDnDgetrf_bufferSize(node.cusolverDnHandle,
                                                node.nrows, node.nrows,
                                                d_AAT, node.nrows,
                                                (int *)&bufferSize));

    // allocate memory for computation
    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }
    checkCudaErrors(cudaMalloc((void **)&d_ipiv, sizeof(int) * node.nrows));

    // set I matrix
    h_I = (double *)calloc(node.nrows * node.nrows, sizeof(double));
    for (i = 0; i < node.nrows; ++i)
    {
        h_I[node.nrows * i + i] = 1.0;
    }
    // checkCudaErrors(cudaMemcpyAsync(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    // checkCudaErrors(cudaMemcpy(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice));
    free(h_I);

    checkCudaErrors(cusolverDnDgetrs(node.cusolverDnHandle, CUBLAS_OP_N,
                                     node.nrows, node.nrows,
                                     d_AAT, node.nrows,
                                     d_ipiv,
                                     (double *)d_buffer, node.nrows,
                                     &info));

    ///////////////////             COMPUTE s = - mat' * y + obj
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.dS, node.dObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, vecY,
                                            &beta, vecS,
                                            CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, vecY,
                                 &beta, vecS,
                                 CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    ///////////////////             FREE RESOURCES
    checkCudaErrors(cusolverDnDestroyParams(cusolverDnParams));

    checkCudaErrors(cusparseDestroyMatDescr(matDescrGen));
    checkCudaErrors(cusparseDestroyDnMat(AAT_descr));

    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecS));

    checkCudaErrors(cudaFree(d_ipiv));
    checkCudaErrors(cudaFree(d_buffer));

    checkCudaErrors(cudaFree(d_AAT));
    checkCudaErrors(cudaFree(d_matDn));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node)
{
    int i, j;
    int signum = 0;
    double deltaX, deltaS, prod, sumX, sumS;

    gsl_vector *x = NULL;
    gsl_vector *y = NULL;
    gsl_vector *s = NULL;
    gsl_matrix *inv = NULL;
    gsl_matrix *mat = NULL;
    gsl_matrix *tmp = NULL;
    gsl_permutation *perm = NULL;

    x = gsl_vector_alloc((size_t)node.ncols);
    y = gsl_vector_alloc((size_t)node.nrows);
    s = gsl_vector_alloc((size_t)node.ncols);
    inv = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.nrows);
    mat = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.ncols);
    tmp = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.ncols);
    perm = gsl_permutation_alloc((size_t)node.nrows);

    // csr to dense
    for (i = 0; i < node.nrows; ++i)
    {
        for (j = node.hCsrMatOffs->data()[i]; j < node.hCsrMatOffs->data()[i + 1]; ++j)
        {
            mat->data[node.ncols * i + node.hCsrMatInds->data()[j]] = node.hCsrMatVals->data()[j];
        }
    }

    ///////////////////             MATRIX MULT
    mat->size1 = node.nrows;
    mat->size2 = node.ncols;
    mat->tda = node.ncols;
    tmp->size1 = node.nrows;
    tmp->size2 = node.nrows;
    tmp->tda = node.ncols;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mat, mat, 0.0, tmp);

    ///////////////////             MATRIX INVERSION

    inv->size1 = node.nrows;
    inv->size2 = node.nrows;
    inv->tda = node.nrows;
    gsl_linalg_LU_decomp(tmp, perm, &signum);
    gsl_linalg_LU_invert(tmp, perm, inv);

    ///////////////////             COMPUTE x = mat' * AAT_inv * rhs

    tmp->size1 = node.ncols;
    tmp->size2 = node.nrows;
    tmp->tda = node.nrows;
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mat, inv, 0.0, tmp);

    // put RHS in Y
    memcpy(y->data, node.hRhsDns, sizeof(double) * node.nrows);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, y, 0.0, x);

    ///////////////////             COMPUTE y = AAT_inv * mat * obj

    tmp->size1 = node.nrows;
    tmp->size2 = node.ncols;
    tmp->tda = node.ncols;

    // put OBJ in S
    memcpy(s->data, node.hObjDns, sizeof(double) * node.ncols);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, mat, 0.0, tmp);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, s, 0.0, y);

    ///////////////////             COMPUTE s = - mat' * y + obj
    gsl_blas_dgemv(CblasTrans, -1.0, mat, y, 1.0, s);

    deltaX = gsl_max(-1.5 * gsl_vector_min(x), 0.0);
    deltaS = gsl_max(-1.5 * gsl_vector_min(s), 0.0);

    gsl_vector_add_constant(x, deltaX);
    gsl_vector_add_constant(s, deltaS);

    gsl_blas_ddot(x, s, &prod);
    prod *= 0.5;

    sumX = 0.0;
    sumS = 0.0;
    for (j = 0; j < node.ncols; ++j)
    {
        sumX += x->data[j];
        sumS += s->data[j];
    }
    deltaX = prod / sumS;
    deltaS = prod / sumX;

    gsl_vector_add_constant(x, deltaX);
    gsl_vector_add_constant(s, deltaS);

    memcpy(node.hX, x->data, sizeof(double) * node.ncols);
    memcpy(node.hY, y->data, sizeof(double) * node.nrows);
    memcpy(node.hS, s->data, sizeof(double) * node.ncols);

    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(s);
    gsl_matrix_free(inv);
    gsl_matrix_free(mat);
    gsl_matrix_free(tmp);
    gsl_permutation_free(perm);

    return CODE_SUCCESFULL;
}


/**
 * Dense Mehrotra interior point solver (stub).
 * The active solver is the sparse path (SyphaNodeSparse / solver_sparse_mehrotra).
 */
SyphaStatus solver_dense_mehrotra(SyphaNodeDense &node)
{
    int iterations = 0;
    double mu = 0.0;

    solver_dense_mehrotra_init(node);

    while ((iterations < node.env->mehrotraMaxIter) && (mu > node.env->mehrotraMuTol))
    {
        ++iterations;
    }

    return CODE_SUCCESFULL;
}

SyphaStatus solver_dense_mehrotra_init(SyphaNodeDense &node)
{
    (void)node;
    return CODE_SUCCESFULL;
}
