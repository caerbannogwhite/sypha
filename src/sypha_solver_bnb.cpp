#include "sypha_solver_bnb.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "sypha_cuda_helper.h"

DeviceNodeWindow::DeviceNodeWindow(int capacity)
{
    cap = (capacity > 0) ? capacity : 1;
    checkCudaErrors(cudaMalloc((void **)&dEntries, sizeof(DeviceQueueEntry) * (size_t)cap));
}

DeviceNodeWindow::~DeviceNodeWindow()
{
    if (dEntries != NULL)
    {
        checkCudaErrors(cudaFree(dEntries));
        dEntries = NULL;
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

std::vector<int> collect_fractional_candidates(const std::vector<double> &x, int ncolsOriginal, double tol)
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
