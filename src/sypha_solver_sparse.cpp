
#include "sypha_solver_sparse.h"
#include "sypha_node_sparse.h"
#include <cctype>
#include <sstream>

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

    free(h_csrAInds);
    free(h_csrAOffs);
    free(h_csrAVals);

    ///////////////////             INITIALISE RHS

    node.env->logger("RHS initialised", "INFO", 17);
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
        checkCudaErrors(cudaMemcpy(result->primalSolution.data(), d_x,
                                   sizeof(double) * node.ncols, cudaMemcpyDeviceToHost));
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

    return infeasibleOrNumerical ? CODE_GENERIC_ERROR : CODE_SUCCESFULL;
}

namespace
{
struct BranchDecision
{
    int varIndex;
    int fixValue; // 0 => x_j <= 0, 1 => x_j >= 1
};

struct BranchNodeState
{
    std::vector<BranchDecision> decisions;
    int depth = 0;
};

struct BaseRelaxationModel
{
    int nrows = 0;
    int ncols = 0;
    int ncolsOriginal = 0;
    int nnz = 0;
    std::vector<int> csrInds;
    std::vector<int> csrOffs;
    std::vector<double> csrVals;
    std::vector<double> obj;
    std::vector<double> rhs;
};

struct IntegerHeuristicResult
{
    bool feasible = false;
    double objective = std::numeric_limits<double>::infinity();
    std::vector<double> solution;
    std::string name;
};

struct DeviceQueueEntry
{
    int nodeId;
    int depth;
    int lastVar;
    int lastFixValue;
};

class DeviceNodeWindow
{
public:
    explicit DeviceNodeWindow(int capacity)
    {
        cap = (capacity > 0) ? capacity : 1;
        checkCudaErrors(cudaMalloc((void **)&dEntries, sizeof(DeviceQueueEntry) * (size_t)cap));
    }

    ~DeviceNodeWindow()
    {
        if (dEntries != NULL)
        {
            checkCudaErrors(cudaFree(dEntries));
            dEntries = NULL;
        }
    }

    bool hasBufferedNode() const
    {
        return cursor < hostWindow.size();
    }

    bool refill(std::deque<int> &frontier, const std::vector<BranchNodeState> &states)
    {
        hostWindow.clear();
        cursor = 0;

        const int fillCount = std::min((int)frontier.size(), cap);
        hostWindow.reserve((size_t)fillCount);

        for (int i = 0; i < fillCount; ++i)
        {
            const int nodeId = frontier.front();
            frontier.pop_front();

            DeviceQueueEntry entry;
            entry.nodeId = nodeId;
            entry.depth = states[(size_t)nodeId].depth;
            if (states[(size_t)nodeId].decisions.empty())
            {
                entry.lastVar = -1;
                entry.lastFixValue = -1;
            }
            else
            {
                const BranchDecision &last = states[(size_t)nodeId].decisions.back();
                entry.lastVar = last.varIndex;
                entry.lastFixValue = last.fixValue;
            }
            hostWindow.push_back(entry);
        }

        if (!hostWindow.empty())
        {
            checkCudaErrors(cudaMemcpy(dEntries, hostWindow.data(),
                                       sizeof(DeviceQueueEntry) * hostWindow.size(),
                                       cudaMemcpyHostToDevice));
        }
        return !hostWindow.empty();
    }

    bool pop(DeviceQueueEntry *outEntry)
    {
        if (!hasBufferedNode())
        {
            return false;
        }

        checkCudaErrors(cudaMemcpy(outEntry, dEntries + cursor,
                                   sizeof(DeviceQueueEntry), cudaMemcpyDeviceToHost));
        ++cursor;
        return true;
    }

private:
    int cap = 1;
    DeviceQueueEntry *dEntries = NULL;
    std::vector<DeviceQueueEntry> hostWindow;
    size_t cursor = 0;
};

class IBranchVariableSelector
{
public:
    virtual ~IBranchVariableSelector() {}
    virtual int select(const std::vector<double> &solution,
                       const std::vector<double> &objective,
                       const std::vector<int> &candidates) const = 0;
};

class IIntegerHeuristic
{
public:
    virtual ~IIntegerHeuristic() {}
    virtual IntegerHeuristicResult tryBuild(const std::vector<double> &relaxedPrimal,
                                            const BaseRelaxationModel &base,
                                            const BranchNodeState &branchNode,
                                            double tol) const = 0;
};

class MostFractionalSelector : public IBranchVariableSelector
{
public:
    int select(const std::vector<double> &solution,
               const std::vector<double> & /*objective*/,
               const std::vector<int> &candidates) const override
    {
        int best = -1;
        double bestScore = -1.0;
        for (int idx : candidates)
        {
            const double frac = fabs(solution[(size_t)idx] - floor(solution[(size_t)idx] + 0.5));
            if (frac > bestScore)
            {
                bestScore = frac;
                best = idx;
            }
        }
        return best;
    }
};

class HighestCostFractionalSelector : public IBranchVariableSelector
{
public:
    int select(const std::vector<double> & /*solution*/,
               const std::vector<double> &objective,
               const std::vector<int> &candidates) const override
    {
        int best = -1;
        double bestCost = -std::numeric_limits<double>::infinity();
        for (int idx : candidates)
        {
            if (objective[(size_t)idx] > bestCost)
            {
                bestCost = objective[(size_t)idx];
                best = idx;
            }
        }
        return best;
    }
};

class NearestIntegerFixingHeuristic : public IIntegerHeuristic
{
public:
    IntegerHeuristicResult tryBuild(const std::vector<double> &relaxedPrimal,
                                    const BaseRelaxationModel &base,
                                    const BranchNodeState &branchNode,
                                    double tol) const override
    {
        IntegerHeuristicResult out;
        out.name = "nearest_integer_fixing";
        out.solution.assign((size_t)base.ncolsOriginal, 0.0);

        if ((int)relaxedPrimal.size() < base.ncolsOriginal)
        {
            return out;
        }

        for (int j = 0; j < base.ncolsOriginal; ++j)
        {
            const double rounded = floor(relaxedPrimal[(size_t)j] + 0.5);
            out.solution[(size_t)j] = rounded < 0.0 ? 0.0 : (rounded > 1.0 ? 1.0 : rounded);
        }

        // Enforce branch decisions in the rounded incumbent.
        for (const BranchDecision &d : branchNode.decisions)
        {
            if (d.varIndex >= 0 && d.varIndex < base.ncolsOriginal)
            {
                out.solution[(size_t)d.varIndex] = (double)d.fixValue;
            }
        }

        // Feasibility check for SCP rows: sum_j A_ij * x_j >= rhs_i
        for (int i = 0; i < base.nrows; ++i)
        {
            double coverage = 0.0;
            for (int k = base.csrOffs[(size_t)i]; k < base.csrOffs[(size_t)i + 1]; ++k)
            {
                const int col = base.csrInds[(size_t)k];
                if (col >= 0 && col < base.ncolsOriginal)
                {
                    coverage += base.csrVals[(size_t)k] * out.solution[(size_t)col];
                }
            }
            if (coverage + tol < base.rhs[(size_t)i])
            {
                return out;
            }
        }

        out.feasible = true;
        out.objective = 0.0;
        for (int j = 0; j < base.ncolsOriginal; ++j)
        {
            out.objective += base.obj[(size_t)j] * out.solution[(size_t)j];
        }
        return out;
    }
};

static std::string to_lower_copy(const std::string &s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c)
                   { return (char)std::tolower(c); });
    return out;
}

static std::unique_ptr<IBranchVariableSelector> make_selector(const std::string &strategy)
{
    const std::string s = to_lower_copy(strategy);
    if (s == "highest_cost_fractional")
    {
        return std::unique_ptr<IBranchVariableSelector>(new HighestCostFractionalSelector());
    }
    return std::unique_ptr<IBranchVariableSelector>(new MostFractionalSelector());
}

static std::vector<std::string> split_csv_tokens(const std::string &csv)
{
    std::vector<std::string> tokens;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        std::string cleaned;
        cleaned.reserve(item.size());
        for (char c : item)
        {
            if (!std::isspace((unsigned char)c))
            {
                cleaned.push_back(c);
            }
        }
        if (!cleaned.empty())
        {
            tokens.push_back(to_lower_copy(cleaned));
        }
    }
    return tokens;
}

static std::vector<std::unique_ptr<IIntegerHeuristic>> make_integer_heuristics(const std::string &configured)
{
    std::vector<std::unique_ptr<IIntegerHeuristic>> heuristics;
    const std::vector<std::string> tokens = split_csv_tokens(configured);
    if (tokens.empty())
    {
        heuristics.push_back(std::unique_ptr<IIntegerHeuristic>(new NearestIntegerFixingHeuristic()));
        return heuristics;
    }

    for (const std::string &token : tokens)
    {
        if (token == "nearest_integer_fixing")
        {
            heuristics.push_back(std::unique_ptr<IIntegerHeuristic>(new NearestIntegerFixingHeuristic()));
        }
    }
    if (heuristics.empty())
    {
        heuristics.push_back(std::unique_ptr<IIntegerHeuristic>(new NearestIntegerFixingHeuristic()));
    }
    return heuristics;
}

static bool append_decision_if_consistent(const BranchNodeState &parent, int var, int value, BranchNodeState *child)
{
    *child = parent;
    for (const BranchDecision &d : parent.decisions)
    {
        if (d.varIndex == var)
        {
            return d.fixValue == value;
        }
    }
    child->decisions.push_back({var, value});
    child->depth = parent.depth + 1;
    return true;
}

static bool is_binary_integral_solution(const std::vector<double> &x, int ncolsOriginal, double tol)
{
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        const double v = x[(size_t)j];
        const double nearest = floor(v + 0.5);
        if (fabs(v - nearest) > tol)
        {
            return false;
        }
        if ((nearest < -tol) || (nearest > 1.0 + tol))
        {
            return false;
        }
    }
    return true;
}

static std::vector<int> collect_fractional_candidates(const std::vector<double> &x, int ncolsOriginal, double tol)
{
    std::vector<int> candidates;
    candidates.reserve((size_t)ncolsOriginal);
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        const double v = x[(size_t)j];
        const double nearest = floor(v + 0.5);
        if ((fabs(v - nearest) > tol) || (nearest < -tol) || (nearest > 1.0 + tol))
        {
            candidates.push_back(j);
        }
    }
    return candidates;
}

static double compute_mip_gap(double incumbent, double dualBound)
{
    if (!std::isfinite(incumbent) || !std::isfinite(dualBound))
    {
        return std::numeric_limits<double>::infinity();
    }
    if (dualBound > incumbent)
    {
        // Invalid for minimization: lower bound cannot exceed incumbent.
        return std::numeric_limits<double>::infinity();
    }
    return (incumbent - dualBound) / std::max(1.0, fabs(incumbent));
}

static void build_branch_model(const BaseRelaxationModel &base,
                               const BranchNodeState &branchNode,
                               std::vector<int> *csrInds,
                               std::vector<int> *csrOffs,
                               std::vector<double> *csrVals,
                               std::vector<double> *obj,
                               std::vector<double> *rhs)
{
    *csrInds = base.csrInds;
    *csrOffs = base.csrOffs;
    *csrVals = base.csrVals;
    *obj = base.obj;
    *rhs = base.rhs;

    const int extraRows = (int)branchNode.decisions.size();
    if (extraRows == 0)
    {
        return;
    }

    csrInds->reserve((size_t)(base.nnz + 2 * extraRows));
    csrVals->reserve((size_t)(base.nnz + 2 * extraRows));
    rhs->reserve((size_t)(base.nrows + extraRows));
    obj->reserve((size_t)(base.ncols + extraRows));
    csrOffs->reserve((size_t)(base.nrows + extraRows + 1));

    for (int row = 0; row < extraRows; ++row)
    {
        const BranchDecision &d = branchNode.decisions[(size_t)row];
        const int slackCol = base.ncols + row;

        csrInds->push_back(d.varIndex);
        csrVals->push_back(d.fixValue == 0 ? -1.0 : 1.0);

        csrInds->push_back(slackCol);
        csrVals->push_back(-1.0);

        csrOffs->push_back(csrOffs->back() + 2);
        rhs->push_back((double)d.fixValue);
        obj->push_back(0.0);
    }
}
} // namespace

SyphaStatus solver_sparse_branch_and_bound(SyphaNodeSparse &node)
{
    BaseRelaxationModel base;
    base.nrows = node.nrows;
    base.ncols = node.ncols;
    base.ncolsOriginal = node.ncolsOriginal;
    base.nnz = node.nnz;
    base.csrInds = *node.hCsrMatInds;
    base.csrOffs = *node.hCsrMatOffs;
    base.csrVals = *node.hCsrMatVals;
    base.obj.assign(node.hObjDns, node.hObjDns + node.ncols);
    base.rhs.assign(node.hRhsDns, node.hRhsDns + node.nrows);

    std::unique_ptr<IBranchVariableSelector> selector = make_selector(node.env->bnbVarSelectionStrategy);
    std::vector<std::unique_ptr<IIntegerHeuristic>> heuristics = make_integer_heuristics(node.env->bnbIntHeuristics);
    const int heuristicFrequency = node.env->bnbHeuristicEveryNNodes > 0 ? node.env->bnbHeuristicEveryNNodes : 10;

    // Keep the original LP matrix resident on the device for the whole BnB run.
    int *d_baseInds = NULL;
    int *d_baseOffs = NULL;
    double *d_baseVals = NULL;
    if (node.dCsrMatInds && node.dCsrMatOffs && node.dCsrMatVals)
    {
        checkCudaErrors(cudaMalloc((void **)&d_baseInds, sizeof(int) * (size_t)base.nnz));
        checkCudaErrors(cudaMalloc((void **)&d_baseOffs, sizeof(int) * (size_t)(base.nrows + 1)));
        checkCudaErrors(cudaMalloc((void **)&d_baseVals, sizeof(double) * (size_t)base.nnz));
        checkCudaErrors(cudaMemcpy(d_baseInds, node.dCsrMatInds, sizeof(int) * (size_t)base.nnz, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_baseOffs, node.dCsrMatOffs, sizeof(int) * (size_t)(base.nrows + 1), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_baseVals, node.dCsrMatVals, sizeof(double) * (size_t)base.nnz, cudaMemcpyDeviceToDevice));
    }

    std::vector<BranchNodeState> nodes;
    nodes.push_back(BranchNodeState());

    std::deque<int> frontier;
    frontier.push_back(0);

    DeviceNodeWindow deviceWindow(node.env->bnbDeviceQueueCapacity);

    const double integralityTol = node.env->bnbIntegralityTol;
    double bestIntegerObj = std::numeric_limits<double>::infinity();
    double globalRelaxLowerBound = std::numeric_limits<double>::infinity();
    std::vector<double> bestIntegerSolution;
    std::string incumbentSource = "none";

    int processedNodes = 0;
    int totalLpIterations = 0;
    char message[1024];
    node.env->logger("Branch-and-bound started", "INFO", 5);
    node.timeSolverStart = node.env->timer();
    const double bnbStartMs = node.timeSolverStart;
    const double logIntervalMs = node.env->bnbLogIntervalSeconds > 0.0 ? node.env->bnbLogIntervalSeconds * 1000.0 : 0.0;
    const double hardTimeLimitMs = node.env->bnbHardTimeLimitSeconds > 0.0 ? node.env->bnbHardTimeLimitSeconds * 1000.0 : 0.0;
    double nextLogMs = bnbStartMs + logIntervalMs;
    bool hardTimeLimitReached = false;
    bool frontierExhausted = false;
    auto releaseBaseCopies = [&]() {
        if (d_baseInds)
            checkCudaErrors(cudaFree(d_baseInds));
        if (d_baseOffs)
            checkCudaErrors(cudaFree(d_baseOffs));
        if (d_baseVals)
            checkCudaErrors(cudaFree(d_baseVals));
        d_baseInds = NULL;
        d_baseOffs = NULL;
        d_baseVals = NULL;
    };

    while (processedNodes < node.env->bnbMaxNodes)
    {
        const double loopNowMs = node.env->timer();
        if ((hardTimeLimitMs > 0.0) && ((loopNowMs - bnbStartMs) >= hardTimeLimitMs))
        {
            hardTimeLimitReached = true;
            node.env->logger("BnB hard time limit reached", "INFO", 5);
            break;
        }
        if ((logIntervalMs > 0.0) && (loopNowMs >= nextLogMs))
        {
            const double currGap = compute_mip_gap(bestIntegerObj, globalRelaxLowerBound);
            if (std::isfinite(currGap))
            {
                sprintf(message, "BnB progress: nodes=%d frontier=%zu incumbent=%.12g dual=%.12g gap=%.4f%% elapsed=%.2fs",
                        processedNodes, frontier.size(), bestIntegerObj, globalRelaxLowerBound, currGap * 100.0, (loopNowMs - bnbStartMs) / 1000.0);
            }
            else
            {
                sprintf(message, "BnB progress: nodes=%d frontier=%zu incumbent=inf dual=%.12g gap=inf elapsed=%.2fs",
                        processedNodes, frontier.size(), globalRelaxLowerBound, (loopNowMs - bnbStartMs) / 1000.0);
            }
            node.env->logger(message, "INFO", 5);
            nextLogMs = loopNowMs + logIntervalMs;
        }

        if (!deviceWindow.hasBufferedNode())
        {
            if (!deviceWindow.refill(frontier, nodes))
            {
                frontierExhausted = true;
                break;
            }
        }

        DeviceQueueEntry entry;
        if (!deviceWindow.pop(&entry))
        {
            continue;
        }

        const BranchNodeState branchNode = nodes[(size_t)entry.nodeId];

        std::vector<int> workInds;
        std::vector<int> workOffs;
        std::vector<double> workVals;
        std::vector<double> workObj;
        std::vector<double> workRhs;
        build_branch_model(base, branchNode, &workInds, &workOffs, &workVals, &workObj, &workRhs);

        node.nrows = base.nrows + (int)branchNode.decisions.size();
        node.ncols = base.ncols + (int)branchNode.decisions.size();
        node.nnz = base.nnz + 2 * (int)branchNode.decisions.size();
        node.ncolsOriginal = base.ncolsOriginal;

        *node.hCsrMatInds = workInds;
        *node.hCsrMatOffs = workOffs;
        *node.hCsrMatVals = workVals;

        if (node.hObjDns)
        {
            free(node.hObjDns);
            node.hObjDns = NULL;
        }
        if (node.hRhsDns)
        {
            free(node.hRhsDns);
            node.hRhsDns = NULL;
        }
        node.hObjDns = (double *)calloc((size_t)node.ncols, sizeof(double));
        node.hRhsDns = (double *)calloc((size_t)node.nrows, sizeof(double));
        memcpy(node.hObjDns, workObj.data(), sizeof(double) * (size_t)node.ncols);
        memcpy(node.hRhsDns, workRhs.data(), sizeof(double) * (size_t)node.nrows);

        if (node.copyModelOnDevice() != CODE_SUCCESFULL)
        {
            releaseBaseCopies();
            return CODE_GENERIC_ERROR;
        }

        SolverExecutionConfig config;
        config.maxIterations = node.env->mehrotraMaxIter;
        config.gapStagnation.enabled = true;
        config.gapStagnation.windowIterations = node.env->bnbGapStallBranchIters;
        config.gapStagnation.minImprovementPct = node.env->bnbGapStallMinImprovPct;

        SolverExecutionResult result;
        const SyphaStatus solveStatus = solver_sparse_mehrotra_run(node, config, &result);
        if (solveStatus != CODE_SUCCESFULL || result.status != CODE_SUCCESFULL)
        {
            if (entry.nodeId == 0)
            {
                node.env->logger("Root LP relaxation infeasible or numerically unstable; aborting BnB", "INFO", 5);
                node.objvalPrim = std::numeric_limits<double>::infinity();
                node.objvalDual = std::numeric_limits<double>::infinity();
                node.mipGap = std::numeric_limits<double>::infinity();
                node.iterations = result.iterations;
                node.timeSolverEnd = node.env->timer();
                releaseBaseCopies();
                return CODE_GENERIC_ERROR;
            }
            continue;
        }

        ++processedNodes;
        totalLpIterations += result.iterations;
        const bool boundIsReliable = (result.terminationReason == SOLVER_TERM_CONVERGED);
        if (boundIsReliable)
        {
            globalRelaxLowerBound = std::min(globalRelaxLowerBound, result.primalObj);
        }

        if ((processedNodes == 1) || ((heuristicFrequency > 0) && (processedNodes % heuristicFrequency == 0)))
        {
            for (const auto &heuristic : heuristics)
            {
                IntegerHeuristicResult heuristicRes = heuristic->tryBuild(result.primalSolution, base, branchNode, integralityTol);
                if (heuristicRes.feasible && heuristicRes.objective < bestIntegerObj - node.env->pxTolerance)
                {
                    bestIntegerObj = heuristicRes.objective;
                    bestIntegerSolution = heuristicRes.solution;
                    incumbentSource = heuristicRes.name;
                    if (node.env->showSolution)
                    {
                        sprintf(message, "New incumbent from heuristic '%s': %.12g", heuristicRes.name.c_str(), heuristicRes.objective);
                    }
                    else
                    {
                        sprintf(message, "New incumbent found: %.12g", heuristicRes.objective);
                    }
                    node.env->logger(message, "INFO", 5);
                    break;
                }
            }
        }

        if (boundIsReliable && (result.primalObj >= bestIntegerObj - node.env->pxTolerance))
        {
            continue;
        }

        const bool isIntegral = is_binary_integral_solution(result.primalSolution, base.ncolsOriginal, integralityTol);
        if (isIntegral)
        {
            bestIntegerObj = result.primalObj;
            bestIntegerSolution = result.primalSolution;
            incumbentSource = "exact_node";
            continue;
        }

        std::vector<int> fractionalCandidates =
            collect_fractional_candidates(result.primalSolution, base.ncolsOriginal, integralityTol);
        if (fractionalCandidates.empty())
        {
            continue;
        }

        const int branchVar = selector->select(result.primalSolution, base.obj, fractionalCandidates);
        if (branchVar < 0)
        {
            continue;
        }

        BranchNodeState childZero;
        if (append_decision_if_consistent(branchNode, branchVar, 0, &childZero))
        {
            nodes.push_back(childZero);
            frontier.push_back((int)nodes.size() - 1);
        }

        BranchNodeState childOne;
        if (append_decision_if_consistent(branchNode, branchVar, 1, &childOne))
        {
            nodes.push_back(childOne);
            frontier.push_back((int)nodes.size() - 1);
        }
    }

    node.iterations = totalLpIterations;
    if (std::isfinite(bestIntegerObj))
    {
        node.objvalPrim = bestIntegerObj;
        if (node.hX)
        {
            free(node.hX);
            node.hX = NULL;
        }
        node.hX = (double *)calloc((size_t)base.ncolsOriginal, sizeof(double));
        memcpy(node.hX, bestIntegerSolution.data(), sizeof(double) * (size_t)base.ncolsOriginal);
    }
    else
    {
        node.objvalPrim = std::numeric_limits<double>::infinity();
    }
    if (std::isfinite(bestIntegerObj) &&
        frontierExhausted &&
        !hardTimeLimitReached &&
        (processedNodes < node.env->bnbMaxNodes))
    {
        // Full tree exhaustion with an incumbent means optimality is proven.
        node.objvalDual = bestIntegerObj;
        node.mipGap = 0.0;
        node.env->logger("Optimality proven: search frontier exhausted", "INFO", 5);
    }
    else
    {
        node.objvalDual = globalRelaxLowerBound;
        node.mipGap = compute_mip_gap(node.objvalPrim, node.objvalDual);
    }

    sprintf(message, "Branch-and-bound processed %d nodes", processedNodes);
    node.env->logger(message, "INFO", 5);
    sprintf(message, "Total LP iterations across nodes: %d", totalLpIterations);
    node.env->logger(message, "INFO", 5);
    if (std::isfinite(bestIntegerObj))
    {
        node.env->logger("Best integer incumbent found", "INFO", 5);
        if (node.env->showSolution)
        {
            sprintf(message, "  Source: %s", incumbentSource.c_str());
            node.env->logger(message, "INFO", 5);
            for (int j = 0; j < base.ncolsOriginal; ++j)
            {
                if (bestIntegerSolution[(size_t)j] > 0.5)
                {
                    sprintf(message, "  x[%d] = %.0f", j, bestIntegerSolution[(size_t)j]);
                    node.env->logger(message, "INFO", 5);
                }
            }
        }
    }
    else
    {
        node.env->logger("No integer incumbent found within node limit", "INFO", 5);
        if (hardTimeLimitReached)
        {
            node.env->logger("Search stopped by hard time limit", "INFO", 5);
        }
        if (node.env->bnbAutoFallbackLp)
        {
            node.env->logger("Falling back to LP relaxation solve", "INFO", 5);

            node.nrows = base.nrows;
            node.ncols = base.ncols;
            node.ncolsOriginal = base.ncolsOriginal;
            node.nnz = base.nnz;

            *node.hCsrMatInds = base.csrInds;
            *node.hCsrMatOffs = base.csrOffs;
            *node.hCsrMatVals = base.csrVals;

            if (node.hObjDns)
            {
                free(node.hObjDns);
                node.hObjDns = NULL;
            }
            if (node.hRhsDns)
            {
                free(node.hRhsDns);
                node.hRhsDns = NULL;
            }
            node.hObjDns = (double *)calloc((size_t)node.ncols, sizeof(double));
            node.hRhsDns = (double *)calloc((size_t)node.nrows, sizeof(double));
            memcpy(node.hObjDns, base.obj.data(), sizeof(double) * (size_t)node.ncols);
            memcpy(node.hRhsDns, base.rhs.data(), sizeof(double) * (size_t)node.nrows);

            if (node.copyModelOnDevice() != CODE_SUCCESFULL)
            {
                releaseBaseCopies();
                return CODE_GENERIC_ERROR;
            }

            releaseBaseCopies();

            const SyphaStatus lpStatus = solver_sparse_mehrotra(node);
            if (lpStatus == CODE_SUCCESFULL)
            {
                node.iterations += totalLpIterations;
            }
            return lpStatus;
        }
    }

    releaseBaseCopies();

    node.timeSolverEnd = node.env->timer();

    return CODE_SUCCESFULL;
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

    node.env->logger("RHS initialised", "INFO", 17);
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
    char message[1024];

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
    node.env->logger("Init: A * A'", "INFO", 20);

    checkCudaErrors(cusparseSpMM(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, matDnDescr,
                                 &beta, AAT_descr,
                                 CUDA_R_64F,
                                 CUSPARSE_CSRMM_ALG1,
                                 d_buffer));

    ///////////////////             MATRIX INVERSION

    node.env->logger("Init: inv(A*A')", "INFO", 20);
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

    sprintf(message, "Init: getrf returned %d", info);
    node.env->logger(message, "INFO", 20);

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

    sprintf(message, "Init: getrs returned %d", info);
    node.env->logger(message, "INFO", 20);

    ///////////////////             COMPUTE s = - mat' * y + obj
    node.env->logger("Init: s = -A'*y + obj", "INFO", 20);
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
    char message[1024];

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
    node.env->logger("Init: A * A'", "INFO", 20);
    mat->size1 = node.nrows;
    mat->size2 = node.ncols;
    mat->tda = node.ncols;
    tmp->size1 = node.nrows;
    tmp->size2 = node.nrows;
    tmp->tda = node.ncols;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mat, mat, 0.0, tmp);

    ///////////////////             MATRIX INVERSION
    node.env->logger("Init: inv(A*A')", "INFO", 20);

    inv->size1 = node.nrows;
    inv->size2 = node.nrows;
    inv->tda = node.nrows;
    gsl_linalg_LU_decomp(tmp, perm, &signum);
    gsl_linalg_LU_invert(tmp, perm, inv);

    ///////////////////             COMPUTE x = mat' * AAT_inv * rhs
    node.env->logger("Init: x = A'*inv(AAT)*rhs", "INFO", 20);

    tmp->size1 = node.ncols;
    tmp->size2 = node.nrows;
    tmp->tda = node.nrows;
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mat, inv, 0.0, tmp);

    // put RHS in Y
    memcpy(y->data, node.hRhsDns, sizeof(double) * node.nrows);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, y, 0.0, x);

    ///////////////////             COMPUTE y = AAT_inv * mat * obj
    node.env->logger("Init: y = inv(AAT)*A*obj", "INFO", 20);

    tmp->size1 = node.nrows;
    tmp->size2 = node.ncols;
    tmp->tda = node.ncols;

    // put OBJ in S
    memcpy(s->data, node.hObjDns, sizeof(double) * node.ncols);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, mat, 0.0, tmp);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, s, 0.0, y);

    ///////////////////             COMPUTE s = - mat' * y + obj
    node.env->logger("Init: s = obj - A'*y", "INFO", 20);
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
