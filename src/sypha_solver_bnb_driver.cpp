#include "sypha_solver_sparse.h"
#include "sypha_solver_bnb.h"
#include "sypha_solver_heuristics.h"
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

SyphaStatus solver_sparse_branch_and_bound(SyphaNodeSparse &node)
{
    char message[1024];
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

    const int originalRowsForPreprocess = node.nrows;
    const int originalColsForPreprocess = node.ncolsOriginal;
    const int originalNnzForPreprocess = node.nnz;

    std::vector<std::unique_ptr<IIntegerHeuristic>> heuristics = makeIntegerHeuristics(node.env->bnbIntHeuristics);
    const int heuristicFrequency = node.env->bnbHeuristicEveryNNodes > 0 ? node.env->bnbHeuristicEveryNNodes : 10;
    const double integralityTol = node.env->bnbIntegralityTol;

    double bestIntegerObj = std::numeric_limits<double>::infinity();
    std::vector<double> bestIntegerSolution;
    std::string incumbentSource = "none";
    double globalRelaxLowerBound = std::numeric_limits<double>::infinity();

    node.env->logger("BnB preprocessing: solving root LP relaxation", "INFO", 5);
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
                bestIntegerSolution = heuristicRes.solution;
                incumbentSource = std::string("presolve_") + heuristicRes.name;
            }
        }

        if (((int)presolveResult.primalSolution.size() >= rootBase.ncolsOriginal) &&
            is_binary_integral_solution(presolveResult.primalSolution, rootBase.ncolsOriginal, integralityTol) &&
            (presolveResult.primalObj < bestIntegerObj - node.env->pxTolerance))
        {
            bestIntegerObj = presolveResult.primalObj;
            bestIntegerSolution.assign(presolveResult.primalSolution.begin(),
                                       presolveResult.primalSolution.begin() + rootBase.ncolsOriginal);
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
        node.env->logger("BnB preprocessing: root LP did not converge, continuing without incumbent bound", "INFO", 5);
    }

    const SyphaStatus preprocessStatus = node.preprocessModel(bestIntegerObj);
    if (preprocessStatus != CODE_SUCCESFULL)
    {
        return preprocessStatus;
    }
    sprintf(message,
            "Preprocessing size: original rows=%d cols=%d nnz=%d -> preprocessed rows=%d cols=%d nnz=%d",
            originalRowsForPreprocess, originalColsForPreprocess, originalNnzForPreprocess,
            node.nrows, node.ncolsOriginal, node.nnz);
    node.env->logger(message, "INFO", 5);
    if (std::isfinite(bestIntegerObj))
    {
        sprintf(message, "Preprocessing incumbent from %s: %.12g", incumbentSource.c_str(), bestIntegerObj);
        node.env->logger(message, "INFO", 5);
    }

    if (node.copyModelOnDevice() != CODE_SUCCESFULL)
    {
        return CODE_GENERIC_ERROR;
    }

    BaseRelaxationModel base;
    buildBaseModel(&base);
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
    // The IPM cannot prove LP bounds tighter than muTol. The global dual
    // bound is the *minimum* over several imprecise LP duals, so the MIP
    // gap can slightly exceed muTol even at true optimality. Use 2*muTol
    // to account for this aggregation effect.
    const double mipGapTolerance = 2.0 * node.env->mehrotraMuTol;
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
        if (std::isfinite(bestIntegerObj) && std::isfinite(globalRelaxLowerBound))
        {
            const double currentGap = compute_mip_gap(bestIntegerObj, globalRelaxLowerBound);
            if (std::isfinite(currentGap) && currentGap <= mipGapTolerance)
            {
                gapToleranceReached = true;
                sprintf(message, "MIP gap %.6f%% within LP solver tolerance (%.6f%%); declaring optimal",
                        currentGap * 100.0, mipGapTolerance * 100.0);
                node.env->logger(message, "INFO", 5);
                break;
            }
        }
        if ((logIntervalMs > 0.0) && (loopNowMs >= nextLogMs))
        {
            // Recompute global dual bound from open frontier before logging.
            {
                double newGlobalBound = std::numeric_limits<double>::infinity();
                for (const int fIdx : frontier)
                {
                    newGlobalBound = std::min(newGlobalBound, nodes[(size_t)fIdx].parentDualBound);
                }
                // Also consider nodes buffered in the device window that
                // have already been popped from the frontier.
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
                    // No open nodes remain: dual bound = incumbent.
                    globalRelaxLowerBound = bestIntegerObj;
                }
            }
            char incumbentStr[64];
            char dualStr[64];
            char gapStr[64];
            if (std::isfinite(bestIntegerObj))
            {
                snprintf(incumbentStr, sizeof(incumbentStr), "%.12g", bestIntegerObj);
            }
            else
            {
                snprintf(incumbentStr, sizeof(incumbentStr), "inf");
            }
            if (std::isfinite(globalRelaxLowerBound))
            {
                snprintf(dualStr, sizeof(dualStr), "%.12g", globalRelaxLowerBound);
            }
            else
            {
                snprintf(dualStr, sizeof(dualStr), "inf");
            }
            const double currGap = compute_mip_gap(bestIntegerObj, globalRelaxLowerBound);
            if (std::isfinite(currGap))
            {
                snprintf(gapStr, sizeof(gapStr), "%.4f%%", currGap * 100.0);
            }
            else
            {
                snprintf(gapStr, sizeof(gapStr), "inf");
            }
            snprintf(message, sizeof(message),
                     "  nodes=%4d frontier=%4zu lp_iters=%5d incumbent=%10s dual=%10s gap=%6s elapsed=%.2fs",
                     processedNodes, frontier.size(), totalLpIterations,
                     incumbentStr, dualStr, gapStr, (loopNowMs - bnbStartMs) / 1000.0);
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

        // Prune before LP solve: if parent dual bound already >= incumbent, skip.
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
            releaseBaseCopies();
            return CODE_GENERIC_ERROR;
        }

        SolverExecutionConfig config;
        config.maxIterations = node.env->mehrotraMaxIter;
        config.gapStagnation.enabled = true;
        config.gapStagnation.windowIterations = node.env->bnbGapStallBranchIters;
        config.gapStagnation.minImprovementPct = node.env->bnbGapStallMinImprovPct;
        config.bnbNodeOrdinal = processedNodes + 1;
        config.denseSelectionLogEveryNodes = 10;

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
        // For minimization, dual objective is the valid node lower bound.
        // Only trust it when the node LP converged and dual<=primal (within tolerance).
        const bool dualBoundConsistent =
            std::isfinite(result.dualObj) &&
            std::isfinite(result.primalObj) &&
            (result.dualObj <= result.primalObj + node.env->pxTolerance);
        const bool boundIsReliableForPruning =
            (result.status == CODE_SUCCESFULL) &&
            (result.terminationReason == SOLVER_TERM_CONVERGED) &&
            dualBoundConsistent;
        const double nodeDualBound = (boundIsReliableForPruning) ? result.dualObj : branchNode.parentDualBound;

        if ((processedNodes == 1) || ((heuristicFrequency > 0) && (processedNodes % heuristicFrequency == 0)))
        {
            bool incumbentImproved = false;
            for (const auto &heuristic : heuristics)
            {
                IntegerHeuristicResult heuristicRes = heuristic->tryBuild(result.primalSolution, result.dualSolution, base, branchNode, integralityTol);
                if (heuristicRes.feasible && heuristicRes.objective < bestIntegerObj - node.env->pxTolerance)
                {
                    bestIntegerObj = heuristicRes.objective;
                    bestIntegerSolution = heuristicRes.solution;
                    incumbentSource = heuristicRes.name;
                    incumbentImproved = true;
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
            // Prune frontier nodes dominated by the new incumbent.
            if (incumbentImproved)
            {
                const double pruneBound = bestIntegerObj - node.env->pxTolerance;
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
                    sprintf(message, "Frontier pruned: %zu -> %zu nodes", before, frontier.size());
                    node.env->logger(message, "INFO", 5);
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
                bestIntegerSolution = result.primalSolution;
                incumbentSource = "exact_node";
                // Prune frontier nodes dominated by the new incumbent.
                const double pruneBound = bestIntegerObj - node.env->pxTolerance;
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
                    sprintf(message, "Frontier pruned: %zu -> %zu nodes", before, frontier.size());
                    node.env->logger(message, "INFO", 5);
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
        node.hX = (double *)calloc((size_t)base.ncolsInputOriginal, sizeof(double));
        const int copyCols = std::min(base.ncolsOriginal, (int)bestIntegerSolution.size());
        for (int j = 0; j < copyCols; ++j)
        {
            const int origCol = base.activeToOriginalCol[(size_t)j];
            if (origCol >= 0 && origCol < base.ncolsInputOriginal)
            {
                node.hX[(size_t)origCol] = bestIntegerSolution[(size_t)j];
            }
        }
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
        // Optimality proven: either the frontier is exhausted or the MIP gap
        // has closed to within the LP solver's numerical precision.
        node.objvalDual = bestIntegerObj;
        node.mipGap = 0.0;
        if (!gapToleranceReached)
        {
            node.env->logger("Optimality proven: search frontier exhausted", "INFO", 5);
        }
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
            const int logCols = std::min(base.ncolsOriginal, (int)bestIntegerSolution.size());
            for (int j = 0; j < logCols; ++j)
            {
                if (bestIntegerSolution[(size_t)j] > 0.5)
                {
                    const int origCol = base.activeToOriginalCol[(size_t)j];
                    sprintf(message, "  x[%d] = %.0f", origCol, bestIntegerSolution[(size_t)j]);
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
