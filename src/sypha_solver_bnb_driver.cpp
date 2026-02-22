#include "sypha_solver_sparse.h"
#include "sypha_solver.h"
#include "sypha_solver_bnb.h"
#include "sypha_solver_heuristics.h"
#include "sypha_preprocessor.h"
#include "sypha_node_sparse.h"

#include "sypha_cuda_helper.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Helper: convert a heuristic/LP solution (in active-column space of a base model)
// to input-original column space.
static void adoptIncumbentSolution(
    std::vector<double> &bestIntegerSolution,
    const std::vector<double> &activeSolution,
    int ncolsOriginal,
    int ncolsInputOriginal,
    const std::vector<int> &activeToOriginalCol)
{
    bestIntegerSolution.assign((size_t)ncolsInputOriginal, 0.0);
    const int copyCols = std::min(ncolsOriginal, (int)activeSolution.size());
    for (int j = 0; j < copyCols; ++j)
    {
        if (activeSolution[(size_t)j] > 0.5)
        {
            const int origCol = activeToOriginalCol[(size_t)j];
            if (origCol >= 0 && origCol < ncolsInputOriginal)
            {
                bestIntegerSolution[(size_t)origCol] = 1.0;
            }
        }
    }
}

// Helper: prune frontier nodes dominated by incumbent.
static void pruneFrontier(
    std::deque<int> &frontier,
    const std::vector<BranchNodeState> &nodes,
    double bestIntegerObj,
    double tol,
    SyphaLogger *log)
{
    const double pruneBound = bestIntegerObj - tol;
    size_t before = frontier.size();
    std::deque<int> surviving;
    for (const int fIdx : frontier)
    {
        if (nodes[(size_t)fIdx].parentDualBound < pruneBound)
        {
            surviving.push_back(fIdx);
        }
    }
    frontier.swap(surviving);
    if (frontier.size() < before)
    {
        log->log(LOG_INFO, "Frontier pruned: %zu -> %zu nodes", before, frontier.size());
    }
}

// refreshNodeFromBase is defined as a lambda inside solver_sparse_branch_and_bound
// (requires friend access to SyphaNodeSparse private members).

// Helper: mid-BnB column removal when incumbent improves.
// Returns number of columns removed.
static int midBnbColumnRemoval(
    BaseRelaxationModel &base,
    double bestIntegerObj,
    double tol,
    std::deque<int> &frontier,
    std::vector<BranchNodeState> &nodes,
    DeviceNodeWindow &deviceWindow,
    SyphaLogger *log)
{
    BaseModelReductionResult reduction = reduce_base_model(base, bestIntegerObj, tol);
    if (reduction.columnsRemoved <= 0)
        return 0;

    log->log(LOG_INFO, "Mid-BnB reduction: %d cols removed, %d remaining",
             reduction.columnsRemoved, base.ncolsOriginal);

    // Remap frontier nodes.
    {
        std::deque<int> surviving;
        for (const int fIdx : frontier)
        {
            if (remap_branch_node(nodes[(size_t)fIdx], reduction.oldToNew))
            {
                surviving.push_back(fIdx);
            }
        }
        frontier.swap(surviving);
    }

    // Remap buffered device window nodes (not yet processed).
    for (size_t wi = deviceWindow.cursorPos(); wi < deviceWindow.windowSize(); ++wi)
    {
        const int bufferedId = deviceWindow.peekNodeId(wi);
        if (!remap_branch_node(nodes[(size_t)bufferedId], reduction.oldToNew))
        {
            // Mark infeasible: will be pruned by bound check.
            nodes[(size_t)bufferedId].parentDualBound = std::numeric_limits<double>::infinity();
        }
    }

    return reduction.columnsRemoved;
}

SyphaStatus solver_sparse_branch_and_bound(SyphaNodeSparse &node)
{
    SyphaLogger *log = node.env->getLogger();
    auto buildBaseModel = [&](BaseRelaxationModel *base) {
        base->nrows = node.nrows;
        base->ncols = node.ncols;
        base->ncolsOriginal = node.ncolsOriginal;
        base->ncolsInputOriginal = node.ncolsInputOriginal > 0 ? node.ncolsInputOriginal : node.ncolsOriginal;
        base->nnz = node.nnz;
        base->csrInds = *node.hCsrMatInds;
        base->csrOffs = *node.hCsrMatOffs;
        base->csrVals = *node.hCsrMatVals;
        base->obj.assign(node.hObjDns, node.hObjDns + node.ncols);
        base->rhs.assign(node.hRhsDns, node.hRhsDns + node.nrows);
        if (node.hActiveToInputCols && !node.hActiveToInputCols->empty())
        {
            base->activeToOriginalCol = *node.hActiveToInputCols;
        }
        else
        {
            base->activeToOriginalCol.resize((size_t)base->ncolsOriginal);
            for (int j = 0; j < base->ncolsOriginal; ++j)
            {
                base->activeToOriginalCol[(size_t)j] = j;
            }
        }
    };

    // Lambda to refresh node host data and device copies from base model after mid-BnB reduction.
    // Must be a lambda (not static function) to access SyphaNodeSparse private members via friend.
    auto refreshNodeFromBase = [&](const BaseRelaxationModel &base,
                                   int *&d_baseInds,
                                   int *&d_baseOffs,
                                   double *&d_baseVals) -> SyphaStatus {
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
        node.hObjDns = (double *)calloc((size_t)base.ncols, sizeof(double));
        node.hRhsDns = (double *)calloc((size_t)base.nrows, sizeof(double));
        memcpy(node.hObjDns, base.obj.data(), sizeof(double) * (size_t)base.ncols);
        memcpy(node.hRhsDns, base.rhs.data(), sizeof(double) * (size_t)base.nrows);

        if (node.copyModelOnDevice() != CODE_SUCCESFULL)
        {
            return CODE_GENERIC_ERROR;
        }

        // Refresh device base copies.
        if (d_baseInds)
            checkCudaErrors(cudaFree(d_baseInds));
        if (d_baseOffs)
            checkCudaErrors(cudaFree(d_baseOffs));
        if (d_baseVals)
            checkCudaErrors(cudaFree(d_baseVals));
        d_baseInds = NULL;
        d_baseOffs = NULL;
        d_baseVals = NULL;

        if (node.dCsrMatInds && node.dCsrMatOffs && node.dCsrMatVals)
        {
            checkCudaErrors(cudaMalloc((void **)&d_baseInds, sizeof(int) * (size_t)base.nnz));
            checkCudaErrors(cudaMalloc((void **)&d_baseOffs, sizeof(int) * (size_t)(base.nrows + 1)));
            checkCudaErrors(cudaMalloc((void **)&d_baseVals, sizeof(double) * (size_t)base.nnz));
            checkCudaErrors(cudaMemcpy(d_baseInds, node.dCsrMatInds, sizeof(int) * (size_t)base.nnz, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(d_baseOffs, node.dCsrMatOffs, sizeof(int) * (size_t)(base.nrows + 1), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(d_baseVals, node.dCsrMatVals, sizeof(double) * (size_t)base.nnz, cudaMemcpyDeviceToDevice));
        }

        return CODE_SUCCESFULL;
    };

    const int originalColsForPreprocess = node.ncolsOriginal;
    const int originalNnzForPreprocess = node.nnz;

    // Determine ncolsInputOriginal early for incumbent storage.
    if (node.ncolsInputOriginal <= 0)
        node.ncolsInputOriginal = node.ncolsOriginal;
    const int ncolsInputOriginal = node.ncolsInputOriginal;

    std::vector<std::unique_ptr<IIntegerHeuristic>> heuristics = makeIntegerHeuristics(node.env->bnbIntHeuristics);
    const int heuristicFrequency = node.env->bnbHeuristicEveryNNodes > 0 ? node.env->bnbHeuristicEveryNNodes : 10;
    const double integralityTol = node.env->bnbIntegralityTol;

    double bestIntegerObj = std::numeric_limits<double>::infinity();
    // Stored in input-original column space (size = ncolsInputOriginal).
    std::vector<double> bestIntegerSolution;
    std::string incumbentSource = "none";
    double globalRelaxLowerBound = std::numeric_limits<double>::infinity();

    // ====================================================================
    // Phase 1: Greedy set cover heuristic (CPU, O(n log n + nnz))
    // ====================================================================
    log->log(LOG_INFO, "BnB preprocessing: running greedy set cover heuristic");
    {
        GreedySetCoverResult greedyResult = greedy_set_cover_heuristic(
            node.nrows, node.ncolsOriginal,
            *node.hCsrMatInds, *node.hCsrMatOffs, *node.hCsrMatVals,
            node.hObjDns);
        if (greedyResult.feasible)
        {
            bestIntegerObj = greedyResult.objective;
            bestIntegerSolution.assign((size_t)ncolsInputOriginal, 0.0);
            // Map from current active column space to input-original space.
            for (int col : greedyResult.selectedColumns)
            {
                int inputCol = col;
                if (node.hActiveToInputCols && !node.hActiveToInputCols->empty())
                    inputCol = (*node.hActiveToInputCols)[(size_t)col];
                if (inputCol >= 0 && inputCol < ncolsInputOriginal)
                    bestIntegerSolution[(size_t)inputCol] = 1.0;
            }
            incumbentSource = "greedy_set_cover";
            log->log(LOG_INFO, "Greedy heuristic incumbent: %.12g", bestIntegerObj);
        }
        else
        {
            log->log(LOG_INFO, "Greedy heuristic did not find a feasible cover");
        }
    }

    // ====================================================================
    // Phase 2: First column removal using greedy incumbent
    // ====================================================================
    if (std::isfinite(bestIntegerObj))
    {
        const int colsBefore = node.ncolsOriginal;
        node.reduceByIncumbent(bestIntegerObj);
        if (node.ncolsOriginal < colsBefore)
        {
            log->log(LOG_INFO, "Greedy reduction: cols %d->%d, nnz %d->%d",
                     colsBefore, node.ncolsOriginal,
                     originalNnzForPreprocess, node.nnz);
        }
    }

    // ====================================================================
    // Phase 2.5: Cost-driven pair/triplet column replacement
    // ====================================================================
    {
        const int colsBefore = node.ncolsOriginal;
        node.applyCostDrivenReduction();
        if (node.ncolsOriginal < colsBefore)
        {
            log->log(LOG_INFO, "Cost-driven pair/triplet reduction: cols %d->%d, nnz %d->%d",
                     colsBefore, node.ncolsOriginal,
                     originalNnzForPreprocess, node.nnz);
        }
    }

    // ====================================================================
    // Phase 2.7: Dominance rules on greedy-reduced model
    // ====================================================================
    {
        const int colsBefore = node.ncolsOriginal;
        node.applyDominancePreprocessing();
        if (node.ncolsOriginal < colsBefore)
        {
            log->log(LOG_INFO, "Pre-LP dominance reduction: cols %d->%d, nnz %d->%d",
                     colsBefore, node.ncolsOriginal,
                     originalNnzForPreprocess, node.nnz);
        }
    }

    // ====================================================================
    // Phase 3: Root LP on reduced model
    // ====================================================================
    log->log(LOG_INFO, "BnB preprocessing: solving root LP relaxation");
    if (node.copyModelOnDevice() != CODE_SUCCESFULL)
    {
        return CODE_GENERIC_ERROR;
    }

    SolverExecutionConfig presolveConfig;
    presolveConfig.maxIterations = node.env->mehrotraMaxIter;
    presolveConfig.gapStagnation.enabled = false;
    presolveConfig.bnbNodeOrdinal = 0;
    presolveConfig.denseSelectionLogEveryNodes = 10;

    SolverExecutionResult presolveResult;
    const SyphaStatus presolveStatus = solver_sparse_mehrotra_run(node, presolveConfig, &presolveResult);
    if ((presolveStatus == CODE_SUCCESFULL) && (presolveResult.status == CODE_SUCCESFULL))
    {
        BaseRelaxationModel rootBase;
        buildBaseModel(&rootBase);
        BranchNodeState rootNode;
        for (const auto &heuristic : heuristics)
        {
            IntegerHeuristicResult heuristicRes = heuristic->tryBuild(
                presolveResult.primalSolution, presolveResult.dualSolution, rootBase, rootNode, integralityTol);
            if (heuristicRes.feasible && heuristicRes.objective < bestIntegerObj - node.env->pxTolerance)
            {
                bestIntegerObj = heuristicRes.objective;
                adoptIncumbentSolution(bestIntegerSolution, heuristicRes.solution,
                                       rootBase.ncolsOriginal, ncolsInputOriginal, rootBase.activeToOriginalCol);
                incumbentSource = std::string("presolve_") + heuristicRes.name;
            }
        }

        if (((int)presolveResult.primalSolution.size() >= rootBase.ncolsOriginal) &&
            is_binary_integral_solution(presolveResult.primalSolution, rootBase.ncolsOriginal, integralityTol) &&
            (presolveResult.primalObj < bestIntegerObj - node.env->pxTolerance))
        {
            bestIntegerObj = presolveResult.primalObj;
            adoptIncumbentSolution(bestIntegerSolution, presolveResult.primalSolution,
                                   rootBase.ncolsOriginal, ncolsInputOriginal, rootBase.activeToOriginalCol);
            incumbentSource = "presolve_exact_root_lp";
        }

        const bool dualBoundConsistent =
            std::isfinite(presolveResult.dualObj) &&
            std::isfinite(presolveResult.primalObj) &&
            (presolveResult.dualObj <= presolveResult.primalObj + node.env->pxTolerance);
        if ((presolveResult.terminationReason == SOLVER_TERM_CONVERGED) && dualBoundConsistent)
        {
            globalRelaxLowerBound = std::min(globalRelaxLowerBound, presolveResult.dualObj);
        }
    }
    else
    {
        log->log(LOG_INFO, "Root LP did not converge, continuing without incumbent bound");
    }

    // ====================================================================
    // Phase 4: Second column removal with improved incumbent
    // ====================================================================
    if (std::isfinite(bestIntegerObj))
    {
        const int colsBefore = node.ncolsOriginal;
        node.reduceByIncumbent(bestIntegerObj);
        if (node.ncolsOriginal < colsBefore)
        {
            log->log(LOG_INFO, "LP heuristic reduction: cols %d->%d", colsBefore, node.ncolsOriginal);
        }
    }

    // ====================================================================
    // Phase 5: Dominance rules on further-reduced model
    // ====================================================================
    node.applyDominancePreprocessing();

    log->log(LOG_INFO, "Preprocessing: cols %d->%d, nnz %d->%d",
             originalColsForPreprocess, node.ncolsOriginal,
             originalNnzForPreprocess, node.nnz);
    if (std::isfinite(bestIntegerObj))
    {
        log->log(LOG_INFO, "Preprocessing incumbent from %s: %.12g", incumbentSource.c_str(), bestIntegerObj);
    }

    // ====================================================================
    // Phase 6: Build base model for BnB
    // ====================================================================
    if (node.copyModelOnDevice() != CODE_SUCCESFULL)
    {
        return CODE_GENERIC_ERROR;
    }

    BaseRelaxationModel base;
    buildBaseModel(&base);

    // Pre-allocate IPM workspace for B&B reuse (avoids ~20 cudaMalloc/cudaFree per node).
    IpmWorkspace ipmWorkspace;
    {
        const int maxBranchDecisions = base.ncolsOriginal;
        const int maxNcols = base.ncols + maxBranchDecisions;
        const int maxNrows = base.nrows + maxBranchDecisions;
        const int maxNnz = base.nnz + 2 * maxBranchDecisions;
        const int maxKktNrows = 2 * maxNcols + maxNrows;
        const int maxKktNnz = 2 * maxNnz + 3 * maxNcols;
        initializeIpmWorkspace(&ipmWorkspace, maxKktNrows, maxKktNnz, maxNcols);
    }

    std::unique_ptr<IBranchVariableSelector> selector = makeBranchSelector(node.env->bnbVarSelectionStrategy);

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
    {
        BranchNodeState rootState;
        rootState.parentDualBound = std::isfinite(globalRelaxLowerBound)
                                        ? globalRelaxLowerBound
                                        : -std::numeric_limits<double>::infinity();
        nodes.push_back(rootState);
    }

    std::deque<int> frontier;
    frontier.push_back(0);

    DeviceNodeWindow deviceWindow(node.env->bnbDeviceQueueCapacity);

    int processedNodes = 0;
    int totalLpIterations = 0;
    bool gapToleranceReached = false;
    const double mipGapTolerance = 2.0 * node.env->mehrotraMuTol;
    log->log(LOG_INFO, "Branch-and-bound started");
    node.timeSolverStart = node.env->timer();
    const double bnbStartMs = node.timeSolverStart;
    const double logIntervalMs = node.env->bnbLogIntervalSeconds > 0.0 ? node.env->bnbLogIntervalSeconds * 1000.0 : 0.0;
    const double hardTimeLimitMs = node.env->bnbHardTimeLimitSeconds > 0.0 ? node.env->bnbHardTimeLimitSeconds * 1000.0 : 0.0;
    double nextLogMs = bnbStartMs + logIntervalMs;
    bool hardTimeLimitReached = false;
    bool frontierExhausted = false;

    // Adaptive iteration control: reduce LP iterations when MIP gap stagnates.
    const int gapStagnationWindow = node.env->bnbGapStagnationWindow;
    const int fullMaxIter = node.env->mehrotraMaxIter;
    const int reducedMaxIter = std::max(5, fullMaxIter / 3);
    double bestMipGapSeen = std::numeric_limits<double>::infinity();
    int nodeAtLastGapImprovement = 0;
    bool iterationsReduced = false;

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

    // ====================================================================
    // BnB main loop
    // ====================================================================
    while (processedNodes < node.env->bnbMaxNodes)
    {
        const double loopNowMs = node.env->timer();
        if ((hardTimeLimitMs > 0.0) && ((loopNowMs - bnbStartMs) >= hardTimeLimitMs))
        {
            hardTimeLimitReached = true;
            log->log(LOG_INFO, "BnB hard time limit reached");
            break;
        }
        if (log->isStopRequested())
        {
            hardTimeLimitReached = true;
            log->log(LOG_INFO, "BnB stopped by watchdog time limit");
            break;
        }
        if (std::isfinite(bestIntegerObj) && std::isfinite(globalRelaxLowerBound))
        {
            const double currentGap = compute_mip_gap(bestIntegerObj, globalRelaxLowerBound);
            if (std::isfinite(currentGap) && currentGap <= mipGapTolerance)
            {
                gapToleranceReached = true;
                log->log(LOG_INFO, "MIP gap %.6f%% within LP solver tolerance (%.6f%%); declaring optimal",
                        currentGap * 100.0, mipGapTolerance * 100.0);
                break;
            }
        }
        if ((logIntervalMs > 0.0) && (loopNowMs >= nextLogMs))
        {
            {
                double newGlobalBound = std::numeric_limits<double>::infinity();
                for (const int fIdx : frontier)
                {
                    newGlobalBound = std::min(newGlobalBound, nodes[(size_t)fIdx].parentDualBound);
                }
                for (size_t wi = deviceWindow.cursorPos(); wi < deviceWindow.windowSize(); ++wi)
                {
                    const int bufferedNodeId = deviceWindow.peekNodeId(wi);
                    newGlobalBound = std::min(newGlobalBound, nodes[(size_t)bufferedNodeId].parentDualBound);
                }
                if (std::isfinite(newGlobalBound))
                {
                    globalRelaxLowerBound = newGlobalBound;
                }
                else if (frontier.empty() && !deviceWindow.hasBufferedNode())
                {
                    globalRelaxLowerBound = bestIntegerObj;
                }
            }
            char incumbentStr[64];
            char dualStr[64];
            char gapStr[64];
            if (std::isfinite(bestIntegerObj))
                snprintf(incumbentStr, sizeof(incumbentStr), "%.12g", bestIntegerObj);
            else
                snprintf(incumbentStr, sizeof(incumbentStr), "inf");
            if (std::isfinite(globalRelaxLowerBound))
                snprintf(dualStr, sizeof(dualStr), "%.12g", globalRelaxLowerBound);
            else
                snprintf(dualStr, sizeof(dualStr), "inf");
            const double currGap = compute_mip_gap(bestIntegerObj, globalRelaxLowerBound);
            if (std::isfinite(currGap))
                snprintf(gapStr, sizeof(gapStr), "%.4f%%", currGap * 100.0);
            else
                snprintf(gapStr, sizeof(gapStr), "inf");
            log->log(LOG_INFO,
                     "  nodes=%4d frontier=%4zu lp_iters=%5d incumbent=%10s dual=%10s gap=%6s elapsed=%.2fs",
                     processedNodes, frontier.size(), totalLpIterations,
                     incumbentStr, dualStr, gapStr, (loopNowMs - bnbStartMs) / 1000.0);
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

        if (branchNode.parentDualBound >= bestIntegerObj - node.env->pxTolerance)
        {
            continue;
        }

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
            releaseIpmWorkspace(&ipmWorkspace);
            releaseBaseCopies();
            return CODE_GENERIC_ERROR;
        }

        SolverExecutionConfig config;
        config.maxIterations = iterationsReduced ? reducedMaxIter : fullMaxIter;
        config.gapStagnation.enabled = true;
        config.gapStagnation.windowIterations = node.env->bnbGapStallBranchIters;
        config.gapStagnation.minImprovementPct = node.env->bnbGapStallMinImprovPct;
        config.bnbNodeOrdinal = processedNodes + 1;
        config.denseSelectionLogEveryNodes = 10;
        config.skipGpuMemorySampling = true;

        SolverExecutionResult result;
        const SyphaStatus solveStatus = solver_sparse_mehrotra_run(node, config, &result, &ipmWorkspace);
        if (solveStatus != CODE_SUCCESFULL || result.status != CODE_SUCCESFULL)
        {
            if (entry.nodeId == 0)
            {
                log->log(LOG_INFO, "Root LP infeasible or numerically unstable; aborting BnB");
                node.objvalPrim = std::numeric_limits<double>::infinity();
                node.objvalDual = std::numeric_limits<double>::infinity();
                node.mipGap = std::numeric_limits<double>::infinity();
                node.iterations = result.iterations;
                node.timeSolverEnd = node.env->timer();
                releaseIpmWorkspace(&ipmWorkspace);
                releaseBaseCopies();
                return CODE_GENERIC_ERROR;
            }
            continue;
        }

        ++processedNodes;
        totalLpIterations += result.iterations;
        const bool dualBoundConsistent =
            std::isfinite(result.dualObj) &&
            std::isfinite(result.primalObj) &&
            (result.dualObj <= result.primalObj + node.env->pxTolerance);
        const bool boundIsReliableForPruning =
            (result.status == CODE_SUCCESFULL) &&
            (result.terminationReason == SOLVER_TERM_CONVERGED) &&
            dualBoundConsistent;
        const double nodeDualBound = (boundIsReliableForPruning) ? result.dualObj : branchNode.parentDualBound;

        const bool dualImproved = boundIsReliableForPruning &&
            (nodeDualBound > branchNode.parentDualBound + node.env->pxTolerance);

        if ((processedNodes == 1) ||
            ((heuristicFrequency > 0) && (processedNodes % heuristicFrequency == 0)) ||
            dualImproved)
        {
            bool incumbentImproved = false;
            for (const auto &heuristic : heuristics)
            {
                IntegerHeuristicResult heuristicRes = heuristic->tryBuild(result.primalSolution, result.dualSolution, base, branchNode, integralityTol);
                if (heuristicRes.feasible && heuristicRes.objective < bestIntegerObj - node.env->pxTolerance)
                {
                    bestIntegerObj = heuristicRes.objective;
                    adoptIncumbentSolution(bestIntegerSolution, heuristicRes.solution,
                                           base.ncolsOriginal, ncolsInputOriginal, base.activeToOriginalCol);
                    incumbentSource = heuristicRes.name;
                    incumbentImproved = true;
                    if (node.env->showSolution)
                        log->log(LOG_INFO, "New incumbent from heuristic '%s': %.12g", heuristicRes.name.c_str(), heuristicRes.objective);
                    else
                        log->log(LOG_INFO, "New incumbent found: %.12g", heuristicRes.objective);
                    break;
                }
            }
            if (incumbentImproved)
            {
                nodeAtLastGapImprovement = processedNodes;
                pruneFrontier(frontier, nodes, bestIntegerObj, node.env->pxTolerance, log);

                // Mid-BnB column removal.
                if (midBnbColumnRemoval(base, bestIntegerObj, node.env->pxTolerance,
                                        frontier, nodes, deviceWindow, log) > 0)
                {
                    if (refreshNodeFromBase(base, d_baseInds, d_baseOffs, d_baseVals) != CODE_SUCCESFULL)
                    {
                        releaseIpmWorkspace(&ipmWorkspace);
                        releaseBaseCopies();
                        return CODE_GENERIC_ERROR;
                    }
                }
            }
        }

        if (nodeDualBound >= bestIntegerObj - node.env->pxTolerance)
        {
            continue;
        }

        const bool isIntegral = is_binary_integral_solution(result.primalSolution, base.ncolsOriginal, integralityTol);
        if (isIntegral)
        {
            if (result.primalObj < bestIntegerObj - node.env->pxTolerance)
            {
                bestIntegerObj = result.primalObj;
                nodeAtLastGapImprovement = processedNodes;
                adoptIncumbentSolution(bestIntegerSolution, result.primalSolution,
                                       base.ncolsOriginal, ncolsInputOriginal, base.activeToOriginalCol);
                incumbentSource = "exact_node";

                pruneFrontier(frontier, nodes, bestIntegerObj, node.env->pxTolerance, log);

                // Mid-BnB column removal.
                if (midBnbColumnRemoval(base, bestIntegerObj, node.env->pxTolerance,
                                        frontier, nodes, deviceWindow, log) > 0)
                {
                    if (refreshNodeFromBase(base, d_baseInds, d_baseOffs, d_baseVals) != CODE_SUCCESFULL)
                    {
                        releaseIpmWorkspace(&ipmWorkspace);
                        releaseBaseCopies();
                        return CODE_GENERIC_ERROR;
                    }
                }
            }
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
            childZero.parentDualBound = nodeDualBound;
            nodes.push_back(childZero);
            frontier.push_back((int)nodes.size() - 1);
        }

        BranchNodeState childOne;
        if (append_decision_if_consistent(branchNode, branchVar, 1, &childOne))
        {
            childOne.parentDualBound = nodeDualBound;
            nodes.push_back(childOne);
            frontier.push_back((int)nodes.size() - 1);
        }

        // Adaptive iteration control: check for MIP gap stagnation.
        if (gapStagnationWindow > 0 && std::isfinite(bestIntegerObj))
        {
            // Periodically recompute global dual bound for accurate gap tracking.
            const int refreshInterval = std::max(1, gapStagnationWindow / 5);
            if (processedNodes % refreshInterval == 0)
            {
                double newGlobalBound = std::numeric_limits<double>::infinity();
                for (const int fIdx : frontier)
                    newGlobalBound = std::min(newGlobalBound, nodes[(size_t)fIdx].parentDualBound);
                for (size_t wi = deviceWindow.cursorPos(); wi < deviceWindow.windowSize(); ++wi)
                    newGlobalBound = std::min(newGlobalBound, nodes[(size_t)deviceWindow.peekNodeId(wi)].parentDualBound);
                if (std::isfinite(newGlobalBound))
                    globalRelaxLowerBound = newGlobalBound;
            }

            const double currentGap = compute_mip_gap(bestIntegerObj, globalRelaxLowerBound);
            if (std::isfinite(currentGap) && currentGap < bestMipGapSeen - 1e-8)
            {
                bestMipGapSeen = currentGap;
                nodeAtLastGapImprovement = processedNodes;
                if (iterationsReduced)
                {
                    iterationsReduced = false;
                    log->log(LOG_INFO, "MIP gap improved to %.4f%%, restoring LP iterations (%d)",
                             currentGap * 100.0, fullMaxIter);
                }
            }

            if (!iterationsReduced && (processedNodes - nodeAtLastGapImprovement >= gapStagnationWindow))
            {
                iterationsReduced = true;
                log->log(LOG_INFO, "MIP gap stagnant for %d nodes, reducing LP iterations: %d -> %d",
                         gapStagnationWindow, fullMaxIter, reducedMaxIter);
            }
        }
    }

    // Recompute global dual bound from remaining open nodes.
    {
        double newGlobalBound = std::numeric_limits<double>::infinity();
        for (const int fIdx : frontier)
        {
            newGlobalBound = std::min(newGlobalBound, nodes[(size_t)fIdx].parentDualBound);
        }
        for (size_t wi = deviceWindow.cursorPos(); wi < deviceWindow.windowSize(); ++wi)
        {
            const int bufferedNodeId = deviceWindow.peekNodeId(wi);
            newGlobalBound = std::min(newGlobalBound, nodes[(size_t)bufferedNodeId].parentDualBound);
        }
        if (std::isfinite(newGlobalBound))
        {
            globalRelaxLowerBound = newGlobalBound;
        }
        else if (frontier.empty() && !deviceWindow.hasBufferedNode() && std::isfinite(bestIntegerObj))
        {
            globalRelaxLowerBound = bestIntegerObj;
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
        // bestIntegerSolution is already in input-original column space.
        node.hX = (double *)calloc((size_t)ncolsInputOriginal, sizeof(double));
        const int copyCols = std::min(ncolsInputOriginal, (int)bestIntegerSolution.size());
        memcpy(node.hX, bestIntegerSolution.data(), sizeof(double) * (size_t)copyCols);
    }
    else
    {
        node.objvalPrim = std::numeric_limits<double>::infinity();
    }
    if (std::isfinite(bestIntegerObj) &&
        (frontierExhausted || gapToleranceReached || (frontier.empty() && !deviceWindow.hasBufferedNode())) &&
        !hardTimeLimitReached &&
        (processedNodes < node.env->bnbMaxNodes))
    {
        node.objvalDual = bestIntegerObj;
        node.mipGap = 0.0;
        if (!gapToleranceReached)
        {
            log->log(LOG_INFO, "Optimality proven: search frontier exhausted");
        }
    }
    else
    {
        node.objvalDual = globalRelaxLowerBound;
        node.mipGap = compute_mip_gap(node.objvalPrim, node.objvalDual);
    }

    log->log(LOG_INFO, "BnB processed %d nodes, %d total LP iterations", processedNodes, totalLpIterations);
    if (std::isfinite(bestIntegerObj))
    {
        log->log(LOG_INFO, "Best integer incumbent found");
        if (node.env->showSolution)
        {
            log->log(LOG_INFO, "  Source: %s", incumbentSource.c_str());
            // bestIntegerSolution is in input-original space.
            for (int j = 0; j < ncolsInputOriginal; ++j)
            {
                if (bestIntegerSolution[(size_t)j] > 0.5)
                {
                    log->log(LOG_INFO, "  x[%d] = 1", j);
                }
            }
        }
    }
    else
    {
        log->log(LOG_INFO, "No integer incumbent found within node limit");
        if (hardTimeLimitReached)
        {
            log->log(LOG_INFO, "Search stopped by hard time limit");
        }
        if (node.env->bnbAutoFallbackLp)
        {
            log->log(LOG_INFO, "Falling back to LP relaxation solve");

            if (refreshNodeFromBase(base, d_baseInds, d_baseOffs, d_baseVals) != CODE_SUCCESFULL)
            {
                releaseIpmWorkspace(&ipmWorkspace);
                releaseBaseCopies();
                return CODE_GENERIC_ERROR;
            }

            releaseIpmWorkspace(&ipmWorkspace);
            releaseBaseCopies();

            const SyphaStatus lpStatus = solver_sparse_mehrotra(node);
            if (lpStatus == CODE_SUCCESFULL)
            {
                node.iterations += totalLpIterations;
            }
            return lpStatus;
        }
    }

    releaseIpmWorkspace(&ipmWorkspace);
    releaseBaseCopies();

    node.timeSolverEnd = node.env->timer();

    return CODE_SUCCESFULL;
}
