#ifndef SYPHA_SOLVER_BNB_H
#define SYPHA_SOLVER_BNB_H

#include <cstddef>
#include <deque>
#include <vector>

#include "sypha_solver_heuristics.h"

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
    explicit DeviceNodeWindow(int capacity);
    ~DeviceNodeWindow();

    bool hasBufferedNode() const;
    bool refill(std::deque<int> &frontier, const std::vector<BranchNodeState> &states);
    bool pop(DeviceQueueEntry *outEntry);
    size_t cursorPos() const;
    size_t windowSize() const;
    int peekNodeId(size_t index) const;

private:
    int cap = 1;
    DeviceQueueEntry *dEntries = NULL;
    std::vector<DeviceQueueEntry> hostWindow;
    size_t cursor = 0;
};

bool append_decision_if_consistent(const BranchNodeState &parent, int var, int value, BranchNodeState *child);
bool is_binary_integral_solution(const std::vector<double> &x, int ncolsOriginal, double tol);
std::vector<int> collect_fractional_candidates(const std::vector<double> &x, int ncolsOriginal, double tol);
double compute_mip_gap(double incumbent, double dualBound);
void build_branch_model(const BaseRelaxationModel &base,
                        const BranchNodeState &branchNode,
                        std::vector<int> *csrInds,
                        std::vector<int> *csrOffs,
                        std::vector<double> *csrVals,
                        std::vector<double> *obj,
                        std::vector<double> *rhs);

#endif // SYPHA_SOLVER_BNB_H
