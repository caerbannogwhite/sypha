#ifndef SYPHA_SOLVER_CUTS_H
#define SYPHA_SOLVER_CUTS_H

#include <memory>
#include <vector>

#include "sypha_solver_heuristics.h"

class ICutSeparator
{
public:
    virtual ~ICutSeparator() {}
    virtual std::vector<CutConstraint> separate(
        const std::vector<double> &primalSolution,
        const std::vector<double> &dualSolution,
        const BaseRelaxationModel &base,
        int ncolsOriginal,
        double integralityTol) const = 0;
};

std::vector<std::unique_ptr<ICutSeparator>> makeCutSeparators();

void append_cuts_to_base_model(BaseRelaxationModel &base,
                                const std::vector<CutConstraint> &cuts);

bool remap_cut_constraint(CutConstraint &cut, const std::vector<int> &oldToNew);

#endif // SYPHA_SOLVER_CUTS_H
