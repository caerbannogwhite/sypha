#include "sypha_solver_cuts.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

// Chvátal-Gomory cut from dual-weighted constraint aggregation.
//
// Given dual multipliers u_i >= 0 and constraints sum_j a_ij x_j >= b_i:
//   Aggregated: sum_j (sum_i u_i a_ij) x_j >= sum_i u_i b_i
//   Since x_j in {0,1}: sum_j ceil(sum_i u_i a_ij) x_j >= ceil(sum_i u_i b_i)
//
// We use positive components of the LP dual solution as multipliers.
// This produces a valid inequality that may cut off the fractional LP optimum.
class DualAggregatedCgSeparator : public ICutSeparator
{
public:
    std::vector<CutConstraint> separate(
        const std::vector<double> &primalSolution,
        const std::vector<double> &dualSolution,
        const BaseRelaxationModel &base,
        int ncolsOriginal,
        double tol) const override
    {
        std::vector<CutConstraint> cuts;

        if (static_cast<int>(dualSolution.size()) < base.nrows ||
            static_cast<int>(primalSolution.size()) < ncolsOriginal)
        {
            return cuts;
        }

        // Aggregate using positive dual values
        std::vector<double> aggCoeffs(static_cast<size_t>(ncolsOriginal), 0.0);
        double rhsSum = 0.0;

        for (int i = 0; i < base.nrows; ++i)
        {
            const double u = std::max(0.0, dualSolution[static_cast<size_t>(i)]);
            if (u < tol)
                continue;
            rhsSum += u * base.rhs[static_cast<size_t>(i)];

            for (int k = base.csrOffs[static_cast<size_t>(i)];
                 k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
            {
                const int col = base.csrInds[static_cast<size_t>(k)];
                if (col >= 0 && col < ncolsOriginal)
                {
                    aggCoeffs[static_cast<size_t>(col)] +=
                        u * base.csrVals[static_cast<size_t>(k)];
                }
            }
        }

        // Round up coefficients and RHS (CG rounding)
        const double cutRhs = std::ceil(rhsSum - tol);
        if (cutRhs <= tol)
            return cuts;

        // Check if the fractional part of rhsSum is significant
        const double f0 = rhsSum - std::floor(rhsSum);
        if (f0 < tol || f0 > 1.0 - tol)
            return cuts;

        CutConstraint cut;
        cut.type = "cg_dual_aggregated";
        cut.rhs = cutRhs;

        double lhsVal = 0.0;
        for (int j = 0; j < ncolsOriginal; ++j)
        {
            const double rounded = std::ceil(aggCoeffs[static_cast<size_t>(j)] - tol);
            if (rounded > tol)
            {
                cut.indices.push_back(j);
                cut.values.push_back(rounded);
                lhsVal += rounded * primalSolution[static_cast<size_t>(j)];
            }
        }

        // Only add if violated by current LP solution
        if (lhsVal < cutRhs - tol && !cut.indices.empty())
        {
            cuts.push_back(std::move(cut));
        }

        return cuts;
    }
};

// Row-pair Chvátal-Gomory cuts.
//
// For each pair of rows (i1, i2) with positive dual values, aggregate the
// two constraints, apply CG rounding, and check for violation. This can
// produce tighter cuts than single-constraint or full-aggregation approaches.
class RowPairCgSeparator : public ICutSeparator
{
public:
    std::vector<CutConstraint> separate(
        const std::vector<double> &primalSolution,
        const std::vector<double> &dualSolution,
        const BaseRelaxationModel &base,
        int ncolsOriginal,
        double tol) const override
    {
        std::vector<CutConstraint> cuts;

        if (static_cast<int>(dualSolution.size()) < base.nrows ||
            static_cast<int>(primalSolution.size()) < ncolsOriginal)
        {
            return cuts;
        }

        // Collect rows with significant positive dual values
        std::vector<int> activeRows;
        activeRows.reserve(static_cast<size_t>(base.nrows));
        for (int i = 0; i < base.nrows; ++i)
        {
            if (dualSolution[static_cast<size_t>(i)] > tol)
            {
                activeRows.push_back(i);
            }
        }

        // Sort by descending dual value (most promising pairs first)
        std::sort(activeRows.begin(), activeRows.end(),
                  [&](int a, int b)
                  {
                      return dualSolution[static_cast<size_t>(a)] >
                             dualSolution[static_cast<size_t>(b)];
                  });

        // Limit to top rows to keep complexity manageable
        const int maxActiveRows = std::min(static_cast<int>(activeRows.size()), 40);

        std::vector<double> aggCoeffs(static_cast<size_t>(ncolsOriginal), 0.0);

        for (int ri = 0; ri < maxActiveRows; ++ri)
        {
            for (int rj = ri + 1; rj < maxActiveRows; ++rj)
            {
                const int i1 = activeRows[static_cast<size_t>(ri)];
                const int i2 = activeRows[static_cast<size_t>(rj)];
                const double u1 = dualSolution[static_cast<size_t>(i1)];
                const double u2 = dualSolution[static_cast<size_t>(i2)];

                // Aggregate the two rows
                const double rhsAgg = u1 * base.rhs[static_cast<size_t>(i1)] +
                                      u2 * base.rhs[static_cast<size_t>(i2)];
                const double f0 = rhsAgg - std::floor(rhsAgg);
                if (f0 < tol || f0 > 1.0 - tol)
                    continue;

                const double cutRhs = std::ceil(rhsAgg - tol);
                if (cutRhs <= tol)
                    continue;

                // Compute aggregated coefficients for original columns only
                std::fill(aggCoeffs.begin(), aggCoeffs.end(), 0.0);

                for (int k = base.csrOffs[static_cast<size_t>(i1)];
                     k < base.csrOffs[static_cast<size_t>(i1) + 1]; ++k)
                {
                    const int col = base.csrInds[static_cast<size_t>(k)];
                    if (col >= 0 && col < ncolsOriginal)
                    {
                        aggCoeffs[static_cast<size_t>(col)] +=
                            u1 * base.csrVals[static_cast<size_t>(k)];
                    }
                }
                for (int k = base.csrOffs[static_cast<size_t>(i2)];
                     k < base.csrOffs[static_cast<size_t>(i2) + 1]; ++k)
                {
                    const int col = base.csrInds[static_cast<size_t>(k)];
                    if (col >= 0 && col < ncolsOriginal)
                    {
                        aggCoeffs[static_cast<size_t>(col)] +=
                            u2 * base.csrVals[static_cast<size_t>(k)];
                    }
                }

                // Round up and check violation
                CutConstraint cut;
                cut.type = "cg_row_pair";
                cut.rhs = cutRhs;
                double lhsVal = 0.0;

                for (int j = 0; j < ncolsOriginal; ++j)
                {
                    const double rounded =
                        std::ceil(aggCoeffs[static_cast<size_t>(j)] - tol);
                    if (rounded > tol)
                    {
                        cut.indices.push_back(j);
                        cut.values.push_back(rounded);
                        lhsVal += rounded * primalSolution[static_cast<size_t>(j)];
                    }
                }

                if (lhsVal < cutRhs - tol && !cut.indices.empty())
                {
                    cuts.push_back(std::move(cut));
                    // Limit cuts from this separator
                    if (cuts.size() >= 30)
                        return cuts;
                }
            }
        }

        return cuts;
    }
};

} // namespace

std::vector<std::unique_ptr<ICutSeparator>> makeCutSeparators()
{
    std::vector<std::unique_ptr<ICutSeparator>> separators;
    separators.push_back(std::make_unique<DualAggregatedCgSeparator>());
    separators.push_back(std::make_unique<RowPairCgSeparator>());
    return separators;
}

void append_cuts_to_base_model(BaseRelaxationModel &base,
                                const std::vector<CutConstraint> &cuts)
{
    if (cuts.empty())
        return;

    const int nCuts = static_cast<int>(cuts.size());
    const int oldNcols = base.ncols;

    // Each cut adds one row and one slack column.
    // Cut row in standard form: sum(a_j * x_j) - s_cut = rhs
    // The slack column index is (oldNcols + cutIndex).

    for (int ci = 0; ci < nCuts; ++ci)
    {
        const CutConstraint &cut = cuts[static_cast<size_t>(ci)];
        const int slackCol = oldNcols + ci;

        // Append cut row entries to CSR
        for (size_t k = 0; k < cut.indices.size(); ++k)
        {
            base.csrInds.push_back(cut.indices[k]);
            base.csrVals.push_back(cut.values[k]);
        }
        // Slack variable entry
        base.csrInds.push_back(slackCol);
        base.csrVals.push_back(-1.0);

        base.csrOffs.push_back(static_cast<int>(base.csrVals.size()));
        base.rhs.push_back(cut.rhs);
        base.obj.push_back(0.0); // Slack has zero objective
    }

    base.nrows += nCuts;
    base.ncols += nCuts;
    base.nnz = static_cast<int>(base.csrVals.size());
}

bool remap_cut_constraint(CutConstraint &cut, const std::vector<int> &oldToNew)
{
    std::vector<int> newIndices;
    std::vector<double> newValues;
    newIndices.reserve(cut.indices.size());
    newValues.reserve(cut.values.size());

    for (size_t k = 0; k < cut.indices.size(); ++k)
    {
        const int oldCol = cut.indices[k];
        if (oldCol < 0 || oldCol >= static_cast<int>(oldToNew.size()))
            continue;
        const int newCol = oldToNew[static_cast<size_t>(oldCol)];
        if (newCol < 0)
            continue; // Column removed; coefficient dropped
        newIndices.push_back(newCol);
        newValues.push_back(cut.values[k]);
    }

    cut.indices = std::move(newIndices);
    cut.values = std::move(newValues);
    return true; // Cut is still valid (may be weaker with fewer terms)
}
