#include "sypha_solver.h"
#include "sypha_solver_sparse.h"
#include "sypha_solver_bnb.h"
#include "sypha_solver_heuristics.h"
#include "sypha_solver_krylov.h"
#include "sypha_node_sparse.h"

#include "sypha_cuda_helper.h"
#include <cstdint>
#include <string>

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
    config.maxIterations = node.env->getMehrotraMaxIter();
    config.gapStagnation.enabled = false;
    config.bnbNodeOrdinal = 0;
    config.denseSelectionLogEveryNodes = 1;

    SolverExecutionResult result;
    SyphaStatus status = solver_sparse_mehrotra_run(node, config, &result);
    if (status != CODE_SUCCESSFUL)
    {
        return status;
    }
    return result.status;
}

SyphaStatus solver_sparse_mehrotra_run(SyphaNodeSparse &node, const SolverExecutionConfig &config, SolverExecutionResult *result, IpmWorkspace *workspace)
{
    const int reorder = node.env->getMehrotraReorder();
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    size_t currBufferSize = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = nullptr;
    double *d_bufferS = nullptr;
    double *d_buffer = nullptr;
    double *d_tmp_prim = nullptr;
    double *d_tmp_dual = nullptr;
    double *d_blockmin_prim = nullptr;
    double *d_blockmin_dual = nullptr;
    int nBlocksAlpha = 0;

    cusparseMatDescr_t A_descr = nullptr;
    DenseLinearSolveWorkspace denseLinearWorkspace;
    const bool useWs = (workspace != nullptr) && workspace->isAllocated;

    enum LinearSolvePath { PATH_DENSE_LU, PATH_SPARSE_QR, PATH_KRYLOV_CG };
    LinearSolvePath linearSolvePath = PATH_SPARSE_QR;
    KrylovSolveWorkspace *krylovWs = nullptr;


    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.hX.resize(node.ncols);
    node.hY.resize(node.nrows);
    node.hS.resize(node.ncols);

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

    int *d_csrAInds = nullptr;
    int *d_csrAOffs = nullptr;
    double *d_csrAVals = nullptr;

    double *d_rhs = nullptr;
    double *d_sol = nullptr;
    double *d_prevSol = nullptr;

    std::vector<int> h_csrAInds(static_cast<size_t>(A_nnz), 0);
    std::vector<int> h_csrAOffs(static_cast<size_t>(A_nrows + 1), 0);
    std::vector<double> h_csrAVals(static_cast<size_t>(A_nnz), 0.0);

    if (node.env->getLogger())
        node.env->getLogger()->log(LOG_TRACE, "KKT matrix: %d x %d, %d non-zeros", A_nrows, A_ncols, A_nnz);

    // Instantiate the first group of n rows: O | A' | I
    // Use O(nnz) CSR->CSC scatter instead of O(ncols * nnz) column scan.
    int off = 0;
    {
        std::vector<int> colCount(static_cast<size_t>(node.ncols), 0);
        for (k = 0; k < node.nnz; ++k)
        {
            const int col = node.hCsrMatInds.data()[k];
            if (col >= 0 && col < node.ncols)
                colCount[static_cast<size_t>(col)]++;
        }

        h_csrAOffs[0] = 0;
        for (j = 0; j < node.ncols; ++j)
        {
            h_csrAOffs[j + 1] = h_csrAOffs[j] + colCount[static_cast<size_t>(j)] + 1; // +1 for I block
        }

        std::vector<int> colCursor(static_cast<size_t>(node.ncols), 0);
        for (i = 0; i < node.nrows; ++i)
        {
            for (k = node.hCsrMatOffs.data()[i]; k < node.hCsrMatOffs.data()[i + 1]; ++k)
            {
                j = node.hCsrMatInds.data()[k];
                if (j >= 0 && j < node.ncols)
                {
                    int pos = h_csrAOffs[j] + colCursor[static_cast<size_t>(j)];
                    h_csrAInds[pos] = node.ncols + i;
                    h_csrAVals[pos] = node.hCsrMatVals.data()[k];
                    colCursor[static_cast<size_t>(j)]++;
                }
            }
        }

        for (j = 0; j < node.ncols; ++j)
        {
            int pos = h_csrAOffs[j] + colCursor[static_cast<size_t>(j)];
            h_csrAInds[pos] = node.ncols + node.nrows + j;
            h_csrAVals[pos] = 1.0;
        }
        off = h_csrAOffs[node.ncols];
    }

    // Instantiate the second group of m rows: A | O | O
    for (i = 0; i < node.nrows; ++i)
    {
        h_csrAOffs[node.ncols + i + 1] = h_csrAOffs[node.ncols + i] + (node.hCsrMatOffs.data()[i + 1] - node.hCsrMatOffs.data()[i]);
    }
    memcpy(&h_csrAInds[off], node.hCsrMatInds.data(), sizeof(int) * node.nnz);
    memcpy(&h_csrAVals[off], node.hCsrMatVals.data(), sizeof(double) * node.nnz);
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

    if (useWs)
    {
        d_csrAInds = workspace->d_csrAInds;
        d_csrAOffs = workspace->d_csrAOffs;
        d_csrAVals = workspace->d_csrAVals;
        A_descr = workspace->A_descr;
    }
    else
    {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_csrAInds), sizeof(int) * A_nnz));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_csrAOffs), sizeof(int) * (A_nrows + 1)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_csrAVals), sizeof(double) * A_nnz));
        checkCudaErrors(cusparseCreateMatDescr(&A_descr));
        checkCudaErrors(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));
    }

    checkCudaErrors(cudaMemcpy(d_csrAInds, h_csrAInds.data(), sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAOffs, h_csrAOffs.data(), sizeof(int) * (A_nrows + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAVals, h_csrAVals.data(), sizeof(double) * A_nnz, cudaMemcpyHostToDevice));

    size_t gpuMemFree = 0;
    size_t gpuMemTotal = 0;
    const bool canTrackGpuMem = !config.skipGpuMemorySampling &&
                                 (cudaMemGetInfo(&gpuMemFree, &gpuMemTotal) == cudaSuccess);
    const size_t gpuMemFreeBeforeKktSetup = gpuMemFree;
    size_t gpuMemFreeAfterKktSetup = gpuMemFree;
    size_t gpuMemMinDuringLinearSolve = gpuMemFree;
    int gpuMemSampleCount = 0;

    if ((node.env->getDenseGpuMemoryFractionThreshold() > 0.0) && canTrackGpuMem)
    {
        const double thresholdBytes = node.env->getDenseGpuMemoryFractionThreshold() * static_cast<double>(gpuMemTotal);
        useDenseLinearSolve = static_cast<double>(denseKktBytes) < thresholdBytes;
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
        // Build template: full sparse-to-dense once; subsequent iterations patch diagonals only.
        buildDenseKktTemplate(&denseLinearWorkspace, node.ncols, node.nrows, node.cusparseHandle);
        if (shouldLogDenseSelection && node.env->getLogger())
        {
            node.env->getLogger()->log(LOG_INFO,
                    "Using dense linear solver with incremental KKT update (dense KKT %.2f MB, sparse KKT %.2f MB, threshold %.2f MB)",
                    static_cast<double>(denseKktBytes) / (1024.0 * 1024.0),
                    static_cast<double>(sparseKktBytes) / (1024.0 * 1024.0),
                    (node.env->getDenseGpuMemoryFractionThreshold() * static_cast<double>(gpuMemTotal)) / (1024.0 * 1024.0));
        }
    }
    else
    {
        // cusolverSpDcsrlsvqr performs a full QR factorization per RHS —
        // 2 full factorizations per IPM iteration.  Dense LU (factorize once,
        // solve twice) is significantly faster when GPU memory permits.
        bool canFitDense = false;
        {
            size_t freeNow = 0, totalNow = 0;
            if (cudaMemGetInfo(&freeNow, &totalNow) == cudaSuccess)
                canFitDense = (static_cast<double>(denseKktBytes) * 1.5 < static_cast<double>(freeNow));
        }

        if (canFitDense)
        {
            initializeDenseLinearSolveWorkspace(&denseLinearWorkspace, A_nrows, A_nnz,
                                                d_csrAOffs, d_csrAInds, d_csrAVals,
                                                node.cusparseHandle, node.cusolverDnHandle);
            buildDenseKktTemplate(&denseLinearWorkspace, node.ncols, node.nrows, node.cusparseHandle);
            useDenseLinearSolve = true;
            if (shouldLogDenseSelection && node.env->getLogger())
            {
                node.env->getLogger()->log(LOG_INFO,
                        "Sparse QR -> dense LU factored with incremental KKT (dense KKT %.2f MB, sparse KKT %.2f MB)",
                        static_cast<double>(denseKktBytes) / (1024.0 * 1024.0),
                        static_cast<double>(sparseKktBytes) / (1024.0 * 1024.0));
            }
        }
        else
        {
            if (shouldLogDenseSelection && node.env->getLogger())
            {
                node.env->getLogger()->log(LOG_DEBUG,
                        "Using sparse QR solver (sparse KKT %.2f MB, dense KKT %.2f MB — insufficient GPU memory for dense LU)",
                        static_cast<double>(sparseKktBytes) / (1024.0 * 1024.0),
                        static_cast<double>(denseKktBytes) / (1024.0 * 1024.0));
            }
        }
    }

    // Determine linear solve path (dense LU, Krylov CG, or sparse QR)
    const std::string &lsStrategy = node.env->getLinearSolverStrategy();
    if (useDenseLinearSolve)
    {
        linearSolvePath = PATH_DENSE_LU;
    }
    else if (lsStrategy == "krylov" ||
             (lsStrategy == "auto" && node.nrows < node.ncols))
    {
        linearSolvePath = PATH_KRYLOV_CG;
        if (useWs)
        {
            if (!workspace->krylov)
                workspace->krylov = new KrylovSolveWorkspace();
            krylovWs = workspace->krylov;
        }
        else
        {
            krylovWs = new KrylovSolveWorkspace();
        }
        krylovWs->maxCgIter = node.env->getKrylovMaxCgIter();
        krylovWs->cgTolInitial = node.env->getKrylovCgTolInitial();
        krylovWs->cgTolFinal = node.env->getKrylovCgTolFinal();
        krylovWs->cgTolDecayRate = node.env->getKrylovCgTolDecayRate();
        initializeKrylovWorkspace(krylovWs, node.nrows, node.ncols);
        if (shouldLogDenseSelection && node.env->getLogger())
        {
            node.env->getLogger()->log(LOG_INFO,
                    "Using Krylov CG solver on normal equations (%d x %d, m < n)",
                    node.nrows, node.ncols);
        }
    }
    else
    {
        linearSolvePath = PATH_SPARSE_QR;
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

    ///////////////////             INITIALISE RHS

    if (useWs)
    {
        d_rhs = workspace->d_rhs;
        d_sol = workspace->d_sol;
        d_prevSol = workspace->d_prevSol;
    }
    else
    {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_rhs), sizeof(double) * A_nrows));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sol), sizeof(double) * A_nrows));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_prevSol), sizeof(double) * A_nrows));
    }

    // put x, y, s on device sol as [x, y, s]
    double *d_x = d_prevSol;
    double *d_y = &d_prevSol[node.ncols];
    double *d_s = &d_prevSol[node.ncols + node.nrows];

    double *d_deltaX = d_sol;
    double *d_deltaY = &d_sol[node.ncols];
    double *d_deltaS = &d_sol[node.ncols + node.nrows];

    checkCudaErrors(cudaMemcpy(d_x, node.hX.data(), sizeof(double) * node.ncols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, node.hY.data(), sizeof(double) * node.nrows, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_s, node.hS.data(), sizeof(double) * node.ncols, cudaMemcpyHostToDevice));

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

    checkCudaErrors(cusparseCreateDnVec(&vecX, static_cast<int64_t>(node.ncols), d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, static_cast<int64_t>(node.nrows), d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResC, static_cast<int64_t>(node.ncols), d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, static_cast<int64_t>(node.nrows), d_resB, CUDA_R_64F));

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
    currBufferSize = static_cast<size_t>(sizeof(double) * node.ncols * 2);
    currBufferSize = currBufferSize > bufferSize ? currBufferSize : bufferSize;
    if (useWs)
    {
        if (currBufferSize > workspace->bufferCapacity)
        {
            if (workspace->d_buffer) checkCudaErrors(cudaFree(workspace->d_buffer));
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->d_buffer), currBufferSize));
            workspace->bufferCapacity = currBufferSize;
        }
        d_buffer = workspace->d_buffer;
    }
    else
    {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));
    }

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
        if (useWs)
        {
            if (currBufferSize > workspace->bufferCapacity)
            {
                checkCudaErrors(cudaFree(workspace->d_buffer));
                checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&workspace->d_buffer), currBufferSize));
                workspace->bufferCapacity = currBufferSize;
            }
            d_buffer = workspace->d_buffer;
        }
        else
        {
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), currBufferSize));
        }
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
    double *d_alphaResult = nullptr;
    if (useWs)
    {
        d_tmp_prim = workspace->d_tmp_prim;
        d_tmp_dual = workspace->d_tmp_dual;
        d_blockmin_prim = workspace->d_blockmin_prim;
        d_blockmin_dual = workspace->d_blockmin_dual;
        d_alphaResult = workspace->d_alphaResult;
    }
    else
    {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_tmp_prim), sizeof(double) * node.ncols));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_tmp_dual), sizeof(double) * node.ncols));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_blockmin_prim), sizeof(double) * nBlocksAlpha));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_blockmin_dual), sizeof(double) * nBlocksAlpha));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_alphaResult), sizeof(double) * 2));
    }

    ///////////////////             MAIN LOOP

    if (node.env->getLogger())
        node.env->getLogger()->log(LOG_TRACE, "Mehrotra procedure started");
    node.timeSolverStart = node.env->timer();
    const int maxIterations = config.maxIterations > 0 ? config.maxIterations : node.env->getMehrotraMaxIter();
    const bool gapStagnationEnabled = config.gapStagnation.enabled && config.gapStagnation.windowIterations > 0 && config.gapStagnation.minImprovementPct >= 0.0;
    const double minImprovementRatio = config.gapStagnation.minImprovementPct / 100.0;
    int noGapImprovementIterations = 0;
    double bestRelativeGap = std::numeric_limits<double>::infinity();
    SolverTerminationReason terminationReason = SOLVER_TERM_MAX_ITER;
    bool infeasibleOrNumerical = false;

    while ((iterations < maxIterations) && (mu > node.env->getMehrotraMuTol()))
    {
        if (node.env->getLogger() && node.env->getLogger()->isStopRequested())
        {
            terminationReason = SOLVER_TERM_TIME_LIMIT;
            break;
        }

        // x, s multiplication: -x.*s on device (was host + 3 full-vector PCIe transfers)
        elem_min_mult_dev(d_x, d_s, d_resXS, node.ncols, node.cudaStream);

        // Dense LU path: factorize once, then reuse for both affine and corrector solves.
        // Uses incremental KKT update: restore template + patch S/X diagonals (avoids
        // full sparse-to-dense conversion every iteration).
        // Krylov CG path: solve normal equations A D^2 A^T dy = f with Jacobi-preconditioned CG.
        // Sparse QR fallback: cusolverSpDcsrlsvqr does not expose separate factor/solve,
        // so we call the monolithic solver for each RHS (only used when GPU memory is
        // insufficient for the dense n*n buffer).
        if (linearSolvePath == PATH_DENSE_LU)
        {
            sampleGpuMemory();
            const bool factorOk = denseLinearWorkspace.templateReady
                ? factorizeDenseLinearSystemIncremental(&denseLinearWorkspace,
                                                        d_s, d_x,
                                                        node.cublasHandle, node.cusolverDnHandle,
                                                        node.cudaStream)
                : factorizeDenseLinearSystem(&denseLinearWorkspace,
                                             node.cusparseHandle, node.cusolverDnHandle);
            if (!factorOk)
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
        else if (linearSolvePath == PATH_KRYLOV_CG)
        {
            sampleGpuMemory();
            // D^2 and Jacobi preconditioner: computed once per IPM iteration
            krylovComputeD2(krylovWs, d_x, d_s, node.ncols, node.cudaStream);
            krylovComputeJacobiDiag(krylovWs,
                                    node.dCsrMatOffs, node.dCsrMatInds, node.dCsrMatVals,
                                    node.nrows, node.ncols, node.cudaStream);
            // Build predictor RHS
            krylovBuildNormalEquationsRHS(krylovWs, d_resC, d_resB, d_resXS,
                                         d_x, d_s, node.nrows, node.ncols,
                                         node.matDescr, node.cusparseHandle, node.cudaStream);
            // Adaptive tolerance
            double cgTol = fmax(krylovWs->cgTolFinal,
                                krylovWs->cgTolInitial * pow(krylovWs->cgTolDecayRate, static_cast<double>(iterations)));
            int cgIters = krylovSolveCG(krylovWs, d_deltaY, krylovWs->d_ne_rhs, cgTol,
                                        node.nrows, node.ncols,
                                        node.matDescr, node.cusparseHandle,
                                        node.cublasHandle, node.cudaStream);
            if (cgIters < 0)
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                if (node.env->getLogger())
                    node.env->getLogger()->log(LOG_DEBUG,
                            "Krylov CG predictor failed at IPM iteration %d", iterations);
                break;
            }
            if (node.env->getLogger())
                node.env->getLogger()->log(LOG_TRACE,
                        "Krylov CG predictor: %d CG iterations (tol=%.2e)", cgIters, cgTol);
            // Back-substitution: recover dx, ds
            krylovRecoverDxDs(krylovWs, d_deltaX, d_deltaS, d_deltaY,
                              d_resC, d_resXS, d_x, d_s,
                              node.nrows, node.ncols,
                              node.matDescr, node.cusparseHandle, node.cudaStream);
            sampleGpuMemory();
        }
        else
        {
            sampleGpuMemory();
            checkCudaErrors(cusolverSpDcsrlsvqr(node.cusolverSpHandle,
                                                A_nrows, A_nnz, A_descr,
                                                d_csrAVals, d_csrAOffs, d_csrAInds,
                                                d_rhs,
                                                node.env->getMehrotraCholTol(), reorder,
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
                      d_alphaResult, &alphaMaxPrim, &alphaMaxDual, node.cudaStream);

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
        corrector_rhs_dev(d_deltaX, d_deltaS, sigma, mu, d_bufferX, node.ncols, node.cudaStream);

        alpha = 1.0;
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alpha, d_bufferX, 1, d_resXS, 1));

        // Corrector step: dense LU reuses factors (triangular solve only);
        // Krylov CG reuses D^2 and preconditioner; sparse QR does another full factorization.
        if (linearSolvePath == PATH_DENSE_LU)
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
        else if (linearSolvePath == PATH_KRYLOV_CG)
        {
            sampleGpuMemory();
            // D^2 and Jacobi preconditioner are unchanged — only rebuild RHS
            krylovBuildNormalEquationsRHS(krylovWs, d_resC, d_resB, d_resXS,
                                         d_x, d_s, node.nrows, node.ncols,
                                         node.matDescr, node.cusparseHandle, node.cudaStream);
            double cgTol = fmax(krylovWs->cgTolFinal,
                                krylovWs->cgTolInitial * pow(krylovWs->cgTolDecayRate, static_cast<double>(iterations)));
            int cgIters = krylovSolveCG(krylovWs, d_deltaY, krylovWs->d_ne_rhs, cgTol,
                                        node.nrows, node.ncols,
                                        node.matDescr, node.cusparseHandle,
                                        node.cublasHandle, node.cudaStream);
            if (cgIters < 0)
            {
                infeasibleOrNumerical = true;
                terminationReason = SOLVER_TERM_INFEASIBLE_OR_NUMERICAL;
                if (node.env->getLogger())
                    node.env->getLogger()->log(LOG_DEBUG,
                            "Krylov CG corrector failed at IPM iteration %d", iterations);
                break;
            }
            if (node.env->getLogger())
                node.env->getLogger()->log(LOG_TRACE,
                        "Krylov CG corrector: %d CG iterations (tol=%.2e)", cgIters, cgTol);
            krylovRecoverDxDs(krylovWs, d_deltaX, d_deltaS, d_deltaY,
                              d_resC, d_resXS, d_x, d_s,
                              node.nrows, node.ncols,
                              node.matDescr, node.cusparseHandle, node.cudaStream);
        }
        else
        {
            sampleGpuMemory();
            checkCudaErrors(cusolverSpDcsrlsvqr(node.cusolverSpHandle,
                                                A_nrows, A_nnz, A_descr,
                                                d_csrAVals, d_csrAOffs, d_csrAInds,
                                                d_rhs,
                                                node.env->getMehrotraCholTol(), reorder,
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
                      d_alphaResult, &alphaMaxPrim, &alphaMaxDual, node.cudaStream);

        alphaPrim = gsl_min(1.0, node.env->getMehrotraEta() * alphaMaxPrim);
        alphaDual = gsl_min(1.0, node.env->getMehrotraEta() * alphaMaxDual);

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

        // update x and s on KKT matrix (not needed for Krylov path)
        if (linearSolvePath != PATH_KRYLOV_CG)
        {
            off = node.nnz * 2 + node.ncols;
            checkCudaErrors(cublasDcopy(node.cublasHandle, node.ncols, d_s, 1, &d_csrAVals[off], 2));
            checkCudaErrors(cublasDcopy(node.cublasHandle, node.ncols, d_x, 1, &d_csrAVals[off + 1], 2));
        }

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
        terminationReason = (mu <= node.env->getMehrotraMuTol()) ? SOLVER_TERM_CONVERGED : SOLVER_TERM_MAX_ITER;
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
        if (node.env->getLogger())
            node.env->getLogger()->log(LOG_INFO, "LP relaxation flagged as infeasible or numerically unstable");
    }

    if (canTrackGpuMem && node.env->getLogger())
    {
        const double toMb = 1024.0 * 1024.0;
        const double freeBeforeMb = static_cast<double>(gpuMemFreeBeforeKktSetup) / toMb;
        const double freeAfterSetupMb = static_cast<double>(gpuMemFreeAfterKktSetup) / toMb;
        const double minDuringSolveMb = static_cast<double>(gpuMemMinDuringLinearSolve) / toMb;
        const double setupUsedMb = (double)(gpuMemFreeBeforeKktSetup >= gpuMemFreeAfterKktSetup ? (gpuMemFreeBeforeKktSetup - gpuMemFreeAfterKktSetup) : 0) / toMb;
        const double extraPeakMb = (double)(gpuMemFreeAfterKktSetup >= gpuMemMinDuringLinearSolve ? (gpuMemFreeAfterKktSetup - gpuMemMinDuringLinearSolve) : 0) / toMb;
        node.env->getLogger()->log(LOG_DEBUG,
                "GPU memory (%s): free before=%.2f MB, after setup=%.2f MB, min during=%.2f MB, setup=%.2f MB, peak=%.2f MB, samples=%d",
                (linearSolvePath == PATH_DENSE_LU) ? "dense" : (linearSolvePath == PATH_KRYLOV_CG) ? "krylov" : "sparse",
                freeBeforeMb, freeAfterSetupMb, minDuringSolveMb, setupUsedMb, extraPeakMb, gpuMemSampleCount);
    }

    if (node.env->getLogger())
        node.env->getLogger()->log(LOG_DEBUG, "Mehrotra procedure finished");
    node.timeSolverEnd = node.env->timer();

    if (result != nullptr)
    {
        result->status = infeasibleOrNumerical ? CODE_GENERIC_ERROR : CODE_SUCCESSFUL;
        result->terminationReason = terminationReason;
        result->iterations = iterations;
        result->primalObj = node.objvalPrim;
        result->dualObj = node.objvalDual;
        result->relativeGap = finalRelativeGap;
        result->primalSolution.resize(static_cast<size_t>(node.ncols), 0.0);
        result->dualSolution.resize(static_cast<size_t>(node.nrows), 0.0);
        checkCudaErrors(cudaMemcpy(result->primalSolution.data(), d_x,
                                   sizeof(double) * node.ncols, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(result->dualSolution.data(), d_y,
                                   sizeof(double) * node.nrows, cudaMemcpyDeviceToHost));
    }

    ///////////////////             RELEASE RESOURCES

    if (!useWs)
    {
        checkCudaErrors(cusparseDestroyMatDescr(A_descr));
        checkCudaErrors(cudaFree(d_csrAInds));
        checkCudaErrors(cudaFree(d_csrAOffs));
        checkCudaErrors(cudaFree(d_csrAVals));
        checkCudaErrors(cudaFree(d_rhs));
        checkCudaErrors(cudaFree(d_sol));
        checkCudaErrors(cudaFree(d_prevSol));
    }

    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecResC));
    checkCudaErrors(cusparseDestroyDnVec(vecResB));

    if (node.matTransDescr)
    {
        checkCudaErrors(cusparseDestroySpMat(node.matTransDescr));
        node.matTransDescr = nullptr;
    }

    if (!useWs)
    {
        if (d_tmp_prim) checkCudaErrors(cudaFree(d_tmp_prim));
        if (d_tmp_dual) checkCudaErrors(cudaFree(d_tmp_dual));
        if (d_blockmin_prim) checkCudaErrors(cudaFree(d_blockmin_prim));
        if (d_blockmin_dual) checkCudaErrors(cudaFree(d_blockmin_dual));
        if (d_alphaResult) checkCudaErrors(cudaFree(d_alphaResult));
        if (d_buffer) checkCudaErrors(cudaFree(d_buffer));
    }

    if (denseLinearWorkspace.isEnabled)
    {
        releaseDenseLinearSolveWorkspace(&denseLinearWorkspace);
    }

    if (!useWs && krylovWs)
    {
        releaseKrylovWorkspace(krylovWs);
        delete krylovWs;
        krylovWs = nullptr;
    }

    return infeasibleOrNumerical ? CODE_GENERIC_ERROR : CODE_SUCCESSFUL;
}
