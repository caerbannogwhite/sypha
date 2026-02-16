#include "sypha_solver_sparse.h"
#include "sypha_solver_bnb.h"
#include "sypha_solver_heuristics.h"
#include "sypha_node_sparse.h"

#include "sypha_cuda_helper.h"

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

    std::unique_ptr<IBranchVariableSelector> selector = makeBranchSelector(node.env->bnbVarSelectionStrategy);
    std::vector<std::unique_ptr<IIntegerHeuristic>> heuristics = makeIntegerHeuristics(node.env->bnbIntHeuristics);
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
        if (boundIsReliableForPruning)
        {
            globalRelaxLowerBound = std::min(globalRelaxLowerBound, result.dualObj);
        }

        if ((processedNodes == 1) || ((heuristicFrequency > 0) && (processedNodes % heuristicFrequency == 0)))
        {
            for (const auto &heuristic : heuristics)
            {
                IntegerHeuristicResult heuristicRes = heuristic->tryBuild(result.primalSolution, result.dualSolution, base, branchNode, integralityTol);
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

        if (boundIsReliableForPruning && (result.dualObj >= bestIntegerObj - node.env->pxTolerance))
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
