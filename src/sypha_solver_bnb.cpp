#include "sypha_solver_bnb.h"
#include "sypha_solver_cuts.h"
#include "sypha_preprocessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "sypha_cuda_helper.h"

DeviceNodeWindow::DeviceNodeWindow(int capacity)
{
    cap = (capacity > 0) ? capacity : 1;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dEntries), sizeof(DeviceQueueEntry) * static_cast<size_t>(cap)));
}

DeviceNodeWindow::~DeviceNodeWindow()
{
    if (dEntries != nullptr)
    {
        checkCudaErrors(cudaFree(dEntries));
        dEntries = nullptr;
    }
}

bool DeviceNodeWindow::hasBufferedNode() const
{
    return cursor < hostWindow.size();
}

bool DeviceNodeWindow::refill(std::deque<int> &frontier, const std::vector<BranchNodeState> &states)
{
    hostWindow.clear();
    cursor = 0;

    const int fillCount = std::min(static_cast<int>(frontier.size()), cap);
    hostWindow.reserve(static_cast<size_t>(fillCount));

    for (int i = 0; i < fillCount; ++i)
    {
        const int nodeId = frontier.front();
        frontier.pop_front();

        DeviceQueueEntry entry;
        entry.nodeId = nodeId;
        entry.depth = states[static_cast<size_t>(nodeId)].depth;
        if (states[static_cast<size_t>(nodeId)].decisions.empty())
        {
            entry.lastVar = -1;
            entry.lastFixValue = -1;
        }
        else
        {
            const BranchDecision &last = states[static_cast<size_t>(nodeId)].decisions.back();
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

bool DeviceNodeWindow::pop(DeviceQueueEntry *outEntry)
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

size_t DeviceNodeWindow::cursorPos() const
{
    return cursor;
}

size_t DeviceNodeWindow::windowSize() const
{
    return hostWindow.size();
}

int DeviceNodeWindow::peekNodeId(size_t index) const
{
    return hostWindow[index].nodeId;
}

BaseModelReductionResult reduce_base_model(
    BaseRelaxationModel &base, double incumbentBound, double tol)
{
    BaseModelReductionResult result;
    result.oldToNew.assign(static_cast<size_t>(base.ncolsOriginal), 0);

    std::vector<int> newToOld;
    std::vector<int> newActiveToOriginal;
    int newCol = 0;
    for (int oldCol = 0; oldCol < base.ncolsOriginal; ++oldCol)
    {
        if (base.obj[static_cast<size_t>(oldCol)] + tol >= incumbentBound)
        {
            result.oldToNew[static_cast<size_t>(oldCol)] = -1;
            ++result.columnsRemoved;
        }
        else
        {
            result.oldToNew[static_cast<size_t>(oldCol)] = newCol;
            newToOld.push_back(oldCol);
            newActiveToOriginal.push_back(base.activeToOriginalCol[static_cast<size_t>(oldCol)]);
            ++newCol;
        }
    }

    if (result.columnsRemoved == 0)
        return result;

    const int newNcolsOriginal = newCol;

    // Rebuild objective
    std::vector<double> newObj(static_cast<size_t>(newNcolsOriginal + base.nrows), 0.0);
    for (int j = 0; j < newNcolsOriginal; ++j)
        newObj[static_cast<size_t>(j)] = base.obj[static_cast<size_t>(newToOld[static_cast<size_t>(j)])];

    // Rebuild CSR
    std::vector<int> newCsrInds;
    std::vector<int> newCsrOffs;
    std::vector<double> newCsrVals;
    newCsrOffs.reserve(static_cast<size_t>(base.nrows) + 1);
    newCsrOffs.push_back(0);

    for (int i = 0; i < base.nrows; ++i)
    {
        for (int k = base.csrOffs[static_cast<size_t>(i)]; k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
        {
            const int oldCol2 = base.csrInds[static_cast<size_t>(k)];
            const double val = base.csrVals[static_cast<size_t>(k)];
            if (oldCol2 >= 0 && oldCol2 < base.ncolsOriginal)
            {
                const int mapped = result.oldToNew[static_cast<size_t>(oldCol2)];
                if (mapped >= 0)
                {
                    newCsrInds.push_back(mapped);
                    newCsrVals.push_back(val);
                }
            }
            else if (oldCol2 == base.ncolsOriginal + i)
            {
                // Slack column: remap to newNcolsOriginal + i
                newCsrInds.push_back(newNcolsOriginal + i);
                newCsrVals.push_back(val);
            }
        }
        newCsrOffs.push_back(static_cast<int>(newCsrVals.size()));
    }

    base.ncolsOriginal = newNcolsOriginal;
    base.ncols = newNcolsOriginal + base.nrows;
    base.nnz = static_cast<int>(newCsrVals.size());
    base.csrInds = std::move(newCsrInds);
    base.csrOffs = std::move(newCsrOffs);
    base.csrVals = std::move(newCsrVals);
    base.obj = std::move(newObj);
    base.activeToOriginalCol = std::move(newActiveToOriginal);

    return result;
}

BaseModelReductionResult reduce_base_model_budget_pruning(
    BaseRelaxationModel &base, double incumbentBound, double tol,
    double preprocessTimeLimitSec)
{
    BaseModelReductionResult result;
    result.oldToNew.assign(static_cast<size_t>(base.ncolsOriginal), 0);

    if (base.nrows <= 0 || base.ncolsOriginal <= 0 || !std::isfinite(incumbentBound))
        return result;

    // Build ColumnPreprocessContext from base model.
    ColumnPreprocessContext ctx;
    ctx.nrows = base.nrows;
    ctx.ncols = base.ncolsOriginal;
    ctx.costs.assign(base.obj.begin(), base.obj.begin() + base.ncolsOriginal);
    ctx.active.assign(static_cast<size_t>(ctx.ncols), 1);
    ctx.incumbentBound = incumbentBound;
    if (preprocessTimeLimitSec > 0.0)
    {
        ctx.deadline = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(preprocessTimeLimitSec));
    }

    // Build rowsByColumn from CSR.
    ctx.rowsByColumn.assign(static_cast<size_t>(ctx.ncols), std::vector<int>());
    for (int i = 0; i < base.nrows; ++i)
    {
        const int begin = base.csrOffs[static_cast<size_t>(i)];
        const int end = base.csrOffs[static_cast<size_t>(i) + 1];
        for (int k = begin; k < end; ++k)
        {
            const int col = base.csrInds[static_cast<size_t>(k)];
            const double val = base.csrVals[static_cast<size_t>(k)];
            if (col >= 0 && col < base.ncolsOriginal && val > tol)
            {
                ctx.rowsByColumn[static_cast<size_t>(col)].push_back(i);
            }
        }
    }

    // Run the incumbent budget pruning rule.
    std::vector<std::unique_ptr<IColumnPreprocessRule>> rules = makeColumnPreprocessRules("incumbent_budget_pruning");
    int removedByRules = 0;
    for (const std::unique_ptr<IColumnPreprocessRule> &rule : rules)
    {
        removedByRules += rule->apply(ctx, tol);
    }

    if (removedByRules <= 0)
        return result;

    // Build oldToNew mapping and rebuild base model.
    std::vector<int> newToOld;
    std::vector<int> newActiveToOriginal;
    int newCol = 0;
    for (int oldCol = 0; oldCol < base.ncolsOriginal; ++oldCol)
    {
        if (!ctx.active[static_cast<size_t>(oldCol)])
        {
            result.oldToNew[static_cast<size_t>(oldCol)] = -1;
            ++result.columnsRemoved;
        }
        else
        {
            result.oldToNew[static_cast<size_t>(oldCol)] = newCol;
            newToOld.push_back(oldCol);
            newActiveToOriginal.push_back(base.activeToOriginalCol[static_cast<size_t>(oldCol)]);
            ++newCol;
        }
    }

    const int newNcolsOriginal = newCol;

    // Rebuild objective.
    std::vector<double> newObj(static_cast<size_t>(newNcolsOriginal + base.nrows), 0.0);
    for (int j = 0; j < newNcolsOriginal; ++j)
        newObj[static_cast<size_t>(j)] = base.obj[static_cast<size_t>(newToOld[static_cast<size_t>(j)])];

    // Rebuild CSR.
    std::vector<int> newCsrInds;
    std::vector<int> newCsrOffs;
    std::vector<double> newCsrVals;
    newCsrOffs.reserve(static_cast<size_t>(base.nrows) + 1);
    newCsrOffs.push_back(0);

    for (int i = 0; i < base.nrows; ++i)
    {
        for (int k = base.csrOffs[static_cast<size_t>(i)]; k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
        {
            const int oldCol2 = base.csrInds[static_cast<size_t>(k)];
            const double val = base.csrVals[static_cast<size_t>(k)];
            if (oldCol2 >= 0 && oldCol2 < base.ncolsOriginal)
            {
                const int mapped = result.oldToNew[static_cast<size_t>(oldCol2)];
                if (mapped >= 0)
                {
                    newCsrInds.push_back(mapped);
                    newCsrVals.push_back(val);
                }
            }
            else if (oldCol2 == base.ncolsOriginal + i)
            {
                newCsrInds.push_back(newNcolsOriginal + i);
                newCsrVals.push_back(val);
            }
        }
        newCsrOffs.push_back(static_cast<int>(newCsrVals.size()));
    }

    base.ncolsOriginal = newNcolsOriginal;
    base.ncols = newNcolsOriginal + base.nrows;
    base.nnz = static_cast<int>(newCsrVals.size());
    base.csrInds = std::move(newCsrInds);
    base.csrOffs = std::move(newCsrOffs);
    base.csrVals = std::move(newCsrVals);
    base.obj = std::move(newObj);
    base.activeToOriginalCol = std::move(newActiveToOriginal);

    return result;
}

bool remap_branch_node(BranchNodeState &branchNode, const std::vector<int> &oldToNew)
{
    std::vector<BranchDecision> newDecisions;
    newDecisions.reserve(branchNode.decisions.size());
    for (const BranchDecision &d : branchNode.decisions)
    {
        if (d.varIndex < 0 || d.varIndex >= static_cast<int>(oldToNew.size()))
        {
            continue;
        }
        const int newVar = oldToNew[static_cast<size_t>(d.varIndex)];
        if (newVar < 0)
        {
            // Column was removed
            if (d.fixValue == 1)
            {
                // Fixed to 1 but column is gone → infeasible
                return false;
            }
            // Fixed to 0, column already removed → redundant, drop
            continue;
        }
        newDecisions.push_back({newVar, d.fixValue});
    }
    branchNode.decisions = std::move(newDecisions);

    // Remap node-level cuts
    for (CutConstraint &cut : branchNode.cuts)
    {
        remap_cut_constraint(cut, oldToNew);
    }

    return true;
}

bool append_decision_if_consistent(const BranchNodeState &parent, int var, int value, BranchNodeState *child)
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

bool is_binary_integral_solution(const std::vector<double> &x, int ncolsOriginal, double tol)
{
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        const double v = x[static_cast<size_t>(j)];
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

std::vector<int> collect_fractional_candidates(const std::vector<double> &x, int ncolsOriginal, double tol)
{
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(ncolsOriginal));
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        const double v = x[static_cast<size_t>(j)];
        const double nearest = floor(v + 0.5);
        if ((fabs(v - nearest) > tol) || (nearest < -tol) || (nearest > 1.0 + tol))
        {
            candidates.push_back(j);
        }
    }
    return candidates;
}

bool has_integer_objective(const double *obj, int ncolsOriginal, double tol)
{
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        const double v = obj[j];
        const double nearest = floor(v + 0.5);
        if (fabs(v - nearest) > tol)
        {
            return false;
        }
    }
    return true;
}

double tighten_dual_bound(double bound, double tol)
{
    if (!std::isfinite(bound))
        return bound;
    return ceil(bound - tol);
}

double compute_mip_gap(double incumbent, double dualBound)
{
    if (!std::isfinite(incumbent) || !std::isfinite(dualBound))
    {
        return std::numeric_limits<double>::infinity();
    }
    if (dualBound > incumbent)
    {
        return std::numeric_limits<double>::infinity();
    }
    return (incumbent - dualBound) / std::max(1.0, fabs(incumbent));
}

void build_branch_model(const BaseRelaxationModel &base,
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

    const int nBranch = static_cast<int>(branchNode.decisions.size());
    const int nCuts = static_cast<int>(branchNode.cuts.size());
    const int extraRows = nBranch + nCuts;
    if (extraRows == 0)
    {
        return;
    }

    // Estimate NNZ for node-level cuts (variable-length rows)
    int cutNnz = 0;
    for (const CutConstraint &c : branchNode.cuts)
    {
        cutNnz += static_cast<int>(c.indices.size()) + 1; // +1 for slack
    }

    csrInds->reserve(static_cast<size_t>(base.nnz + 2 * nBranch + cutNnz));
    csrVals->reserve(static_cast<size_t>(base.nnz + 2 * nBranch + cutNnz));
    rhs->reserve(static_cast<size_t>(base.nrows + extraRows));
    obj->reserve(static_cast<size_t>(base.ncols + extraRows));
    csrOffs->reserve(static_cast<size_t>(base.nrows + extraRows + 1));

    // Append branching constraint rows
    for (int row = 0; row < nBranch; ++row)
    {
        const BranchDecision &d = branchNode.decisions[static_cast<size_t>(row)];
        const int slackCol = base.ncols + row;

        csrInds->push_back(d.varIndex);
        csrVals->push_back(d.fixValue == 0 ? -1.0 : 1.0);

        csrInds->push_back(slackCol);
        csrVals->push_back(-1.0);

        csrOffs->push_back(csrOffs->back() + 2);
        rhs->push_back(static_cast<double>(d.fixValue));
        obj->push_back(0.0);
    }

    // Append node-level cut rows
    for (int ci = 0; ci < nCuts; ++ci)
    {
        const CutConstraint &cut = branchNode.cuts[static_cast<size_t>(ci)];
        const int slackCol = base.ncols + nBranch + ci;

        for (size_t k = 0; k < cut.indices.size(); ++k)
        {
            csrInds->push_back(cut.indices[k]);
            csrVals->push_back(cut.values[k]);
        }

        csrInds->push_back(slackCol);
        csrVals->push_back(-1.0);

        csrOffs->push_back(csrOffs->back() +
                           static_cast<int>(cut.indices.size()) + 1);
        rhs->push_back(cut.rhs);
        obj->push_back(0.0);
    }
}
