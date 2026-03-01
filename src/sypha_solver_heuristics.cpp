#include "sypha_solver_heuristics.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace
{
class MostFractionalSelector : public IBranchVariableSelector
{
public:
    int select(const std::vector<double> &solution,
               const std::vector<double> & /*objective*/,
               const std::vector<int> &candidates) const override
    {
        int best = -1;
        double bestScore = -1.0;
        for (int idx : candidates)
        {
            const double frac = fabs(solution[static_cast<size_t>(idx)] - floor(solution[static_cast<size_t>(idx)] + 0.5));
            if (frac > bestScore)
            {
                bestScore = frac;
                best = idx;
            }
        }
        return best;
    }
};

class HighestCostFractionalSelector : public IBranchVariableSelector
{
public:
    int select(const std::vector<double> & /*solution*/,
               const std::vector<double> &objective,
               const std::vector<int> &candidates) const override
    {
        int best = -1;
        double bestCost = -std::numeric_limits<double>::infinity();
        for (int idx : candidates)
        {
            if (objective[static_cast<size_t>(idx)] > bestCost)
            {
                bestCost = objective[static_cast<size_t>(idx)];
                best = idx;
            }
        }
        return best;
    }
};

class NearestIntegerFixingHeuristic : public IIntegerHeuristic
{
public:
    IntegerHeuristicResult tryBuild(const std::vector<double> &relaxedPrimal,
                                    const std::vector<double> & /*relaxedDual*/,
                                    const BaseRelaxationModel &base,
                                    const BranchNodeState &branchNode,
                                    double tol) const override
    {
        IntegerHeuristicResult out;
        out.name = "nearest_integer_fixing";
        out.solution.assign(static_cast<size_t>(base.ncolsOriginal), 0.0);

        if (static_cast<int>(relaxedPrimal.size()) < base.ncolsOriginal)
        {
            return out;
        }

        for (int j = 0; j < base.ncolsOriginal; ++j)
        {
            const double rounded = floor(relaxedPrimal[static_cast<size_t>(j)] + 0.5);
            out.solution[static_cast<size_t>(j)] = rounded < 0.0 ? 0.0 : (rounded > 1.0 ? 1.0 : rounded);
        }

        for (const BranchDecision &d : branchNode.decisions)
        {
            if (d.varIndex >= 0 && d.varIndex < base.ncolsOriginal)
            {
                out.solution[static_cast<size_t>(d.varIndex)] = static_cast<double>(d.fixValue);
            }
        }

        for (int i = 0; i < base.nrows; ++i)
        {
            double coverage = 0.0;
            for (int k = base.csrOffs[static_cast<size_t>(i)]; k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
            {
                const int col = base.csrInds[static_cast<size_t>(k)];
                if (col >= 0 && col < base.ncolsOriginal)
                {
                    coverage += base.csrVals[static_cast<size_t>(k)] * out.solution[static_cast<size_t>(col)];
                }
            }
            if (coverage + tol < base.rhs[static_cast<size_t>(i)])
            {
                return out;
            }
        }

        out.feasible = true;
        out.objective = 0.0;
        for (int j = 0; j < base.ncolsOriginal; ++j)
        {
            out.objective += base.obj[static_cast<size_t>(j)] * out.solution[static_cast<size_t>(j)];
        }
        return out;
    }
};

class DualGuidedCoverRepairHeuristic : public IIntegerHeuristic
{
public:
    IntegerHeuristicResult tryBuild(const std::vector<double> &relaxedPrimal,
                                    const std::vector<double> &relaxedDual,
                                    const BaseRelaxationModel &base,
                                    const BranchNodeState &branchNode,
                                    double tol) const override
    {
        IntegerHeuristicResult out;
        out.name = "dual_guided_cover_repair";
        out.solution.assign(static_cast<size_t>(base.ncolsOriginal), 0.0);

        if (static_cast<int>(relaxedPrimal.size()) < base.ncolsOriginal)
        {
            return out;
        }

        const int nrows = base.nrows;
        const int ncols = base.ncolsOriginal;
        std::vector<char> fixedZero(static_cast<size_t>(ncols), 0);
        std::vector<char> fixedOne(static_cast<size_t>(ncols), 0);
        std::vector<double> coverage(static_cast<size_t>(nrows), 0.0);

        for (const BranchDecision &d : branchNode.decisions)
        {
            if (d.varIndex < 0 || d.varIndex >= ncols)
            {
                continue;
            }
            if (d.fixValue == 0)
            {
                fixedZero[static_cast<size_t>(d.varIndex)] = 1;
            }
            else
            {
                fixedOne[static_cast<size_t>(d.varIndex)] = 1;
                out.solution[static_cast<size_t>(d.varIndex)] = 1.0;
            }
        }

        for (int j = 0; j < ncols; ++j)
        {
            if (fixedZero[static_cast<size_t>(j)])
            {
                out.solution[static_cast<size_t>(j)] = 0.0;
                continue;
            }
            if (fixedOne[static_cast<size_t>(j)])
            {
                continue;
            }
            if (relaxedPrimal[static_cast<size_t>(j)] >= 1.0 - tol)
            {
                out.solution[static_cast<size_t>(j)] = 1.0;
            }
        }

        auto recomputeCoverage = [&]() {
            std::fill(coverage.begin(), coverage.end(), 0.0);
            for (int i = 0; i < nrows; ++i)
            {
                for (int k = base.csrOffs[static_cast<size_t>(i)]; k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
                {
                    const int col = base.csrInds[static_cast<size_t>(k)];
                    if (col >= 0 && col < ncols && out.solution[static_cast<size_t>(col)] > 0.5)
                    {
                        coverage[static_cast<size_t>(i)] += base.csrVals[static_cast<size_t>(k)];
                    }
                }
            }
        };

        recomputeCoverage();

        auto isRowCovered = [&](int row) {
            return coverage[static_cast<size_t>(row)] + tol >= base.rhs[static_cast<size_t>(row)];
        };

        while (true)
        {
            int uncovered = -1;
            for (int i = 0; i < nrows; ++i)
            {
                if (!isRowCovered(i))
                {
                    uncovered = i;
                    break;
                }
            }
            if (uncovered < 0)
            {
                break;
            }

            int bestCol = -1;
            double bestScore = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < ncols; ++j)
            {
                if (out.solution[static_cast<size_t>(j)] > 0.5 || fixedZero[static_cast<size_t>(j)])
                {
                    continue;
                }

                double uncoveredGain = 0.0;
                double dualGain = 0.0;
                for (int i = 0; i < nrows; ++i)
                {
                    if (isRowCovered(i))
                    {
                        continue;
                    }
                    for (int k = base.csrOffs[static_cast<size_t>(i)]; k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
                    {
                        if (base.csrInds[static_cast<size_t>(k)] != j)
                        {
                            continue;
                        }
                        const double aij = base.csrVals[static_cast<size_t>(k)];
                        if (aij > 0.0)
                        {
                            uncoveredGain += aij;
                            if (static_cast<int>(relaxedDual.size()) > i)
                            {
                                dualGain += std::max(0.0, relaxedDual[static_cast<size_t>(i)]) * aij;
                            }
                        }
                        break;
                    }
                }

                if (uncoveredGain <= 0.0)
                {
                    continue;
                }
                const double colCost = std::max(1e-9, base.obj[static_cast<size_t>(j)]);
                const double score = (uncoveredGain + dualGain) / colCost;
                if (score > bestScore)
                {
                    bestScore = score;
                    bestCol = j;
                }
            }

            if (bestCol < 0)
            {
                int fallbackCol = -1;
                double bestFallbackCost = std::numeric_limits<double>::infinity();
                for (int i = 0; i < nrows; ++i)
                {
                    if (isRowCovered(i))
                    {
                        continue;
                    }
                    for (int k = base.csrOffs[static_cast<size_t>(i)]; k < base.csrOffs[static_cast<size_t>(i) + 1]; ++k)
                    {
                        const int col = base.csrInds[static_cast<size_t>(k)];
                        if (col < 0 || col >= ncols || fixedZero[static_cast<size_t>(col)] || out.solution[static_cast<size_t>(col)] > 0.5)
                        {
                            continue;
                        }
                        if (base.csrVals[static_cast<size_t>(k)] <= 0.0)
                        {
                            continue;
                        }
                        if (base.obj[static_cast<size_t>(col)] < bestFallbackCost)
                        {
                            bestFallbackCost = base.obj[static_cast<size_t>(col)];
                            fallbackCol = col;
                        }
                    }
                }
                if (fallbackCol < 0)
                {
                    return out;
                }
                bestCol = fallbackCol;
            }
            out.solution[static_cast<size_t>(bestCol)] = 1.0;
            recomputeCoverage();
        }

        std::vector<int> selected;
        for (int j = 0; j < ncols; ++j)
        {
            if (out.solution[static_cast<size_t>(j)] > 0.5 && !fixedOne[static_cast<size_t>(j)])
            {
                selected.push_back(j);
            }
        }
        std::sort(selected.begin(), selected.end(),
                  [&](int a, int b)
                  { return base.obj[static_cast<size_t>(a)] > base.obj[static_cast<size_t>(b)]; });

        for (int col : selected)
        {
            out.solution[static_cast<size_t>(col)] = 0.0;
            recomputeCoverage();
            bool feasible = true;
            for (int i = 0; i < nrows; ++i)
            {
                if (!isRowCovered(i))
                {
                    feasible = false;
                    break;
                }
            }
            if (!feasible)
            {
                out.solution[static_cast<size_t>(col)] = 1.0;
                recomputeCoverage();
            }
        }

        for (int i = 0; i < nrows; ++i)
        {
            if (!isRowCovered(i))
            {
                return out;
            }
        }

        out.feasible = true;
        out.objective = 0.0;
        for (int j = 0; j < ncols; ++j)
        {
            out.objective += base.obj[static_cast<size_t>(j)] * out.solution[static_cast<size_t>(j)];
        }
        return out;
    }
};

std::string toLowerCopy(const std::string &s)
{
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::vector<std::string> splitCsvTokens(const std::string &csv)
{
    std::vector<std::string> tokens;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        std::string cleaned;
        cleaned.reserve(item.size());
        for (char c : item)
        {
            if (!std::isspace((unsigned char)c))
            {
                cleaned.push_back(c);
            }
        }
        if (!cleaned.empty())
        {
            tokens.push_back(toLowerCopy(cleaned));
        }
    }
    return tokens;
}
} // namespace

std::unique_ptr<IBranchVariableSelector> makeBranchSelector(const std::string &strategy)
{
    const std::string s = toLowerCopy(strategy);
    if (s == "highest_cost_fractional")
    {
        return std::make_unique<HighestCostFractionalSelector>();
    }
    return std::make_unique<MostFractionalSelector>();
}

std::vector<std::unique_ptr<IIntegerHeuristic>> makeIntegerHeuristics(const std::string &configured)
{
    std::vector<std::unique_ptr<IIntegerHeuristic>> heuristics;
    const std::vector<std::string> tokens = splitCsvTokens(configured);
    if (tokens.empty())
    {
        heuristics.push_back(std::make_unique<NearestIntegerFixingHeuristic>());
        heuristics.push_back(std::make_unique<DualGuidedCoverRepairHeuristic>());
        return heuristics;
    }

    for (const std::string &token : tokens)
    {
        if (token == "nearest_integer_fixing")
        {
            heuristics.push_back(std::make_unique<NearestIntegerFixingHeuristic>());
        }
        else if (token == "dual_guided_cover_repair")
        {
            heuristics.push_back(std::make_unique<DualGuidedCoverRepairHeuristic>());
        }
    }
    if (heuristics.empty())
    {
        heuristics.push_back(std::make_unique<NearestIntegerFixingHeuristic>());
        heuristics.push_back(std::make_unique<DualGuidedCoverRepairHeuristic>());
    }
    return heuristics;
}
