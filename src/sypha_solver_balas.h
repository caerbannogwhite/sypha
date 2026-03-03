#ifndef SYPHA_SOLVER_BALAS_H
#define SYPHA_SOLVER_BALAS_H

#include <vector>

#include "sypha_solver_heuristics.h"

// Compute reduced costs: s_j = c_j - sum_i u_i * a_{ij}
std::vector<double> compute_reduced_costs(
    const std::vector<double> &objective,
    const std::vector<double> &dualSolution,
    const BaseRelaxationModel &base,
    int ncolsOriginal);

// BBG result: branching sets R = {R_1, ..., R_p} and decision flag
struct BalasBranchResult
{
    std::vector<std::vector<int>> sets; // R_1 ... R_p
    bool useBR1;                        // true if Algorithm 6 chose BR1
};

// BBG + Algorithm 6: generate branching sets and decide BR1 vs BR2
BalasBranchResult balas_branch_generate(
    const std::vector<double> &primalSolution,
    const std::vector<double> &dualSolution,
    const std::vector<double> &reducedCosts,
    const BaseRelaxationModel &base,
    int ncolsOriginal,
    int maxBranches,
    double integralityTol);

// BR1: create p children with multi-variable fixings + cover cuts
std::vector<BranchNodeState> balas_br1_children(
    const BranchNodeState &parent,
    const std::vector<std::vector<int>> &branchSets,
    double parentDualBound,
    double parentDualBoundRaw);

#endif // SYPHA_SOLVER_BALAS_H
