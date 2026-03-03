#include "sypha_solver_balas.h"
#include "sypha_solver_bnb.h"

#include <algorithm>
#include <cmath>
#include <vector>

std::vector<double> compute_reduced_costs(
    const std::vector<double> &objective,
    const std::vector<double> &dualSolution,
    const BaseRelaxationModel &base,
    int ncolsOriginal)
{
    std::vector<double> s(static_cast<size_t>(ncolsOriginal));
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        s[static_cast<size_t>(j)] = objective[static_cast<size_t>(j)];
    }

    for (int i = 0; i < base.nrows; ++i)
    {
        const double u = dualSolution[static_cast<size_t>(i)];
        if (u <= 0.0)
            continue;

        for (int k = base.csrOffs[static_cast<size_t>(i)];
             k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
        {
            const int col = base.csrInds[static_cast<size_t>(k)];
            if (col >= 0 && col < ncolsOriginal)
            {
                s[static_cast<size_t>(col)] -=
                    u * base.csrVals[static_cast<size_t>(k)];
            }
        }
    }

    return s;
}

BalasBranchResult balas_branch_generate(
    const std::vector<double> &primalSolution,
    const std::vector<double> &dualSolution,
    const std::vector<double> &reducedCosts,
    const BaseRelaxationModel &base,
    int ncolsOriginal,
    int maxBranches,
    double integralityTol)
{
    BalasBranchResult result;
    result.useBR1 = false;

    // Build fractional variable lookup
    std::vector<bool> isFractional(static_cast<size_t>(ncolsOriginal), false);
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        const double v = primalSolution[static_cast<size_t>(j)];
        const double nearest = std::floor(v + 0.5);
        if (std::fabs(v - nearest) > integralityTol)
        {
            isFractional[static_cast<size_t>(j)] = true;
        }
    }

    // For each row with positive dual, collect fractional columns
    // with negative reduced cost
    for (int i = 0; i < base.nrows; ++i)
    {
        if (dualSolution[static_cast<size_t>(i)] <= integralityTol)
            continue;

        std::vector<int> Ri;
        for (int k = base.csrOffs[static_cast<size_t>(i)];
             k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
        {
            const int col = base.csrInds[static_cast<size_t>(k)];
            if (col < 0 || col >= ncolsOriginal)
                continue;
            if (!isFractional[static_cast<size_t>(col)])
                continue;
            if (reducedCosts[static_cast<size_t>(col)] >= -integralityTol)
                continue;
            if (base.csrVals[static_cast<size_t>(k)] <= integralityTol)
                continue;

            Ri.push_back(col);
        }

        if (!Ri.empty())
        {
            std::sort(Ri.begin(), Ri.end());
            result.sets.push_back(std::move(Ri));
        }
    }

    // Deduplicate sets (remove exact duplicates after sorting)
    std::sort(result.sets.begin(), result.sets.end());
    result.sets.erase(std::unique(result.sets.begin(), result.sets.end()),
                      result.sets.end());

    // Algorithm 6 decision
    const int p = static_cast<int>(result.sets.size());
    if (p >= 2)
    {
        int totalVars = 0;
        int singletonCount = 0;
        for (const std::vector<int> &s : result.sets)
        {
            totalVars += static_cast<int>(s.size());
            if (s.size() == 1)
                ++singletonCount;
        }

        result.useBR1 = (totalVars > static_cast<int>(p * std::log2(static_cast<double>(p)))) &&
                         (singletonCount <= 1) &&
                         (p <= maxBranches);
    }

    return result;
}

std::vector<BranchNodeState> balas_br1_children(
    const BranchNodeState &parent,
    const std::vector<std::vector<int>> &branchSets,
    double parentDualBound,
    double parentDualBoundRaw)
{
    const int p = static_cast<int>(branchSets.size());
    std::vector<BranchNodeState> children;
    children.reserve(static_cast<size_t>(p));

    for (int i = 0; i < p; ++i)
    {
        BranchNodeState child = parent;

        // Fix all variables in R_i to 0
        bool consistent = true;
        for (const int j : branchSets[static_cast<size_t>(i)])
        {
            // Check if variable already fixed
            bool alreadyFixed = false;
            for (const BranchDecision &d : child.decisions)
            {
                if (d.varIndex == j)
                {
                    if (d.fixValue != 0)
                        consistent = false;
                    alreadyFixed = true;
                    break;
                }
            }
            if (!consistent)
                break;
            if (!alreadyFixed)
            {
                child.decisions.push_back({j, 0});
            }
        }

        if (!consistent)
            continue;

        // Add cover cuts for each prior set R_k (k < i):
        // sum_{j in R_k} x_j >= 1
        for (int k = 0; k < i; ++k)
        {
            CutConstraint cut;
            cut.type = "balas_cover";
            cut.rhs = 1.0;
            for (const int j : branchSets[static_cast<size_t>(k)])
            {
                cut.indices.push_back(j);
                cut.values.push_back(1.0);
            }
            child.cuts.push_back(std::move(cut));
        }

        child.depth = parent.depth + 1;
        child.parentDualBound = parentDualBound;
        child.parentDualBoundRaw = parentDualBoundRaw;
        children.push_back(std::move(child));
    }

    return children;
}
