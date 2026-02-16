#ifndef SYPHA_SOLVER_HEURISTICS_H
#define SYPHA_SOLVER_HEURISTICS_H

#include <limits>
#include <memory>
#include <string>
#include <vector>

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
    int ncolsOriginal = 0;      // Active original columns after preprocessing
    int ncolsInputOriginal = 0; // Original columns in input instance
    int nnz = 0;
    std::vector<int> csrInds;
    std::vector<int> csrOffs;
    std::vector<double> csrVals;
    std::vector<double> obj;
    std::vector<double> rhs;
    std::vector<int> activeToOriginalCol;
};

struct IntegerHeuristicResult
{
    bool feasible = false;
    double objective = std::numeric_limits<double>::infinity();
    std::vector<double> solution;
    std::string name;
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
                                            const std::vector<double> &relaxedDual,
                                            const BaseRelaxationModel &base,
                                            const BranchNodeState &branchNode,
                                            double tol) const = 0;
};

std::unique_ptr<IBranchVariableSelector> makeBranchSelector(const std::string &strategy);
std::vector<std::unique_ptr<IIntegerHeuristic>> makeIntegerHeuristics(const std::string &configured);

#endif // SYPHA_SOLVER_HEURISTICS_H
