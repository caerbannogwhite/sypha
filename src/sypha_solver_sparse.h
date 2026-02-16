#ifndef SYPHA_SOLVER_SPARSE_H
#define SYPHA_SOLVER_SPARSE_H

#include "common.h"
#include "sypha_solver_utils.h"
#include "sypha_environment_defaults.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"

class SyphaNodeSparse;

enum SolverTerminationReason
{
    SOLVER_TERM_CONVERGED = 0,
    SOLVER_TERM_MAX_ITER,
    SOLVER_TERM_GAP_STALLED,
    SOLVER_TERM_INFEASIBLE_OR_NUMERICAL,
};

struct SolverGapStagnationConfig
{
    bool enabled = false;
    int windowIterations = 0;
    double minImprovementPct = 0.0;
};

struct SolverExecutionConfig
{
    int maxIterations = sypha_environment_defaults::kMehrotraMaxIter;
    SolverGapStagnationConfig gapStagnation;
    int bnbNodeOrdinal = 0;
    int denseSelectionLogEveryNodes = 1;
};

struct SolverExecutionResult
{
    SyphaStatus status = CODE_GENERIC_ERROR;
    SolverTerminationReason terminationReason = SOLVER_TERM_MAX_ITER;
    int iterations = 0;
    double primalObj = 0.0;
    double dualObj = 0.0;
    double relativeGap = std::numeric_limits<double>::infinity();
    std::vector<double> primalSolution;
    std::vector<double> dualSolution;
};

SyphaStatus solver_sparse_mehrotra_run(SyphaNodeSparse &node, const SolverExecutionConfig &config, SolverExecutionResult *result);
SyphaStatus solver_sparse_branch_and_bound(SyphaNodeSparse &node);

#endif // SYPHA_SOLVER_SPARSE_H