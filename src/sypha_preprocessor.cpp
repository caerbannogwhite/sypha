#include "sypha_preprocessor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

GreedySetCoverResult greedy_set_cover_heuristic(
    int nrows,
    int ncolsOriginal,
    const std::vector<int> &csrInds,
    const std::vector<int> &csrOffs,
    const std::vector<double> &csrVals,
    const double *objDns)
{
    GreedySetCoverResult result;
    if (nrows <= 0 || ncolsOriginal <= 0)
    {
        return result;
    }

    // Build rowsByColumn (transpose CSR row→col).
    std::vector<std::vector<int>> rowsByColumn(static_cast<size_t>(ncolsOriginal));
    for (int i = 0; i < nrows; ++i)
    {
        const int begin = csrOffs[static_cast<size_t>(i)];
        const int end = csrOffs[static_cast<size_t>(i) + 1];
        for (int k = begin; k < end; ++k)
        {
            const int col = csrInds[static_cast<size_t>(k)];
            if (col >= 0 && col < ncolsOriginal && csrVals[static_cast<size_t>(k)] > 0.0)
            {
                rowsByColumn[static_cast<size_t>(col)].push_back(i);
            }
        }
    }

    // Build sortable list: (cost, -coverageCount, colIndex).
    struct ColEntry
    {
        double cost;
        int negCoverage;
        int colIndex;
    };
    std::vector<ColEntry> columns(static_cast<size_t>(ncolsOriginal));
    for (int j = 0; j < ncolsOriginal; ++j)
    {
        columns[static_cast<size_t>(j)] = {objDns[j], -static_cast<int>(rowsByColumn[static_cast<size_t>(j)].size()), j};
    }
    std::sort(columns.begin(), columns.end(), [](const ColEntry &a, const ColEntry &b) {
        if (a.cost != b.cost)
            return a.cost < b.cost;
        return a.negCoverage < b.negCoverage; // more coverage first
    });

    // Greedy scan: include a column if it covers at least one uncovered row.
    std::vector<char> covered(static_cast<size_t>(nrows), 0);
    int uncoveredCount = nrows;
    double totalCost = 0.0;

    for (const ColEntry &entry : columns)
    {
        if (uncoveredCount <= 0)
            break;

        int newCoverage = 0;
        for (int row : rowsByColumn[static_cast<size_t>(entry.colIndex)])
        {
            if (!covered[static_cast<size_t>(row)])
                ++newCoverage;
        }

        if (newCoverage > 0)
        {
            for (int row : rowsByColumn[static_cast<size_t>(entry.colIndex)])
            {
                if (!covered[static_cast<size_t>(row)])
                {
                    covered[static_cast<size_t>(row)] = 1;
                    --uncoveredCount;
                }
            }
            totalCost += entry.cost;
            result.selectedColumns.push_back(entry.colIndex);
        }
    }

    if (uncoveredCount == 0)
    {
        result.feasible = true;
        result.objective = totalCost;
    }

    return result;
}

namespace
{
bool isSubsetSorted(const std::vector<int> &subset, const std::vector<int> &superset)
{
    size_t i = 0;
    size_t j = 0;
    while (i < subset.size() && j < superset.size())
    {
        if (subset[i] == superset[j])
        {
            ++i;
            ++j;
        }
        else if (subset[i] > superset[j])
        {
            ++j;
        }
        else
        {
            return false;
        }
    }
    return i == subset.size();
}

bool unionCoversSorted(const std::vector<int> &target,
                       const std::vector<int> &a,
                       const std::vector<int> &b);

bool tripleUnionCoversSorted(const std::vector<int> &target,
                             const std::vector<int> &a,
                             const std::vector<int> &b,
                             const std::vector<int> &c)
{
    size_t ia = 0, ib = 0, ic = 0;
    for (int row : target)
    {
        while (ia < a.size() && a[ia] < row)
            ++ia;
        if (ia < a.size() && a[ia] == row)
            continue;
        while (ib < b.size() && b[ib] < row)
            ++ib;
        if (ib < b.size() && b[ib] == row)
            continue;
        while (ic < c.size() && c[ic] < row)
            ++ic;
        if (ic < c.size() && c[ic] == row)
            continue;
        return false;
    }
    return true;
}

bool unionCoversSorted(const std::vector<int> &target,
                       const std::vector<int> &a,
                       const std::vector<int> &b)
{
    size_t ia = 0;
    size_t ib = 0;
    for (int row : target)
    {
        while (ia < a.size() && a[ia] < row)
        {
            ++ia;
        }
        const bool inA = (ia < a.size() && a[ia] == row);
        if (inA)
        {
            continue;
        }
        while (ib < b.size() && b[ib] < row)
        {
            ++ib;
        }
        const bool inB = (ib < b.size() && b[ib] == row);
        if (!inB)
        {
            return false;
        }
    }
    return true;
}

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

class SingleColumnDominanceRule : public IColumnPreprocessRule
{
public:
    const char *name() const override
    {
        return "single_column_dominance";
    }

    int apply(ColumnPreprocessContext &ctx, double tol) const override
    {
        int removed = 0;
        for (int target = 0; target < ctx.ncols; ++target)
        {
            if (std::chrono::steady_clock::now() >= ctx.deadline)
                break;
            if (!ctx.active[static_cast<size_t>(target)])
            {
                continue;
            }
            const std::vector<int> &targetRows = ctx.rowsByColumn[static_cast<size_t>(target)];
            for (int cand = 0; cand < ctx.ncols; ++cand)
            {
                if (cand == target || !ctx.active[static_cast<size_t>(cand)])
                {
                    continue;
                }
                if (ctx.costs[static_cast<size_t>(cand)] > ctx.costs[static_cast<size_t>(target)] + tol)
                {
                    continue;
                }

                const std::vector<int> &candRows = ctx.rowsByColumn[static_cast<size_t>(cand)];
                if (!isSubsetSorted(targetRows, candRows))
                {
                    continue;
                }

                if (fabs(ctx.costs[static_cast<size_t>(cand)] - ctx.costs[static_cast<size_t>(target)]) <= tol && cand > target)
                {
                    continue;
                }

                ctx.active[static_cast<size_t>(target)] = 0;
                ++removed;
                break;
            }
        }
        return removed;
    }
};

class TwoColumnDominanceRule : public IColumnPreprocessRule
{
public:
    const char *name() const override
    {
        return "two_column_dominance";
    }

    int apply(ColumnPreprocessContext &ctx, double tol) const override
    {
        int removed = 0;
        bool timedOut = false;
        for (int target = 0; target < ctx.ncols && !timedOut; ++target)
        {
            if (std::chrono::steady_clock::now() >= ctx.deadline)
                break;
            if (!ctx.active[static_cast<size_t>(target)])
            {
                continue;
            }

            const std::vector<int> &targetRows = ctx.rowsByColumn[static_cast<size_t>(target)];
            const double targetCost = ctx.costs[static_cast<size_t>(target)];
            bool dominated = false;

            for (int a = 0; a < ctx.ncols && !dominated; ++a)
            {
                if ((a & 255) == 0 && std::chrono::steady_clock::now() >= ctx.deadline)
                {
                    timedOut = true;
                    break;
                }
                if (a == target || !ctx.active[static_cast<size_t>(a)])
                {
                    continue;
                }
                const double aCost = ctx.costs[static_cast<size_t>(a)];
                if (aCost >= targetCost - tol)
                {
                    continue;
                }

                for (int b = a + 1; b < ctx.ncols; ++b)
                {
                    if (b == target || !ctx.active[static_cast<size_t>(b)])
                    {
                        continue;
                    }
                    const double pairCost = aCost + ctx.costs[static_cast<size_t>(b)];
                    if (pairCost >= targetCost - tol)
                    {
                        continue;
                    }
                    if (unionCoversSorted(targetRows, ctx.rowsByColumn[static_cast<size_t>(a)], ctx.rowsByColumn[static_cast<size_t>(b)]))
                    {
                        dominated = true;
                        break;
                    }
                }
            }

            if (dominated)
            {
                ctx.active[static_cast<size_t>(target)] = 0;
                ++removed;
            }
        }
        return removed;
    }
};
class CostDrivenReplacementRule : public IColumnPreprocessRule
{
public:
    const char *name() const override
    {
        return "cost_driven_replacement";
    }

    int apply(ColumnPreprocessContext &ctx, double tol) const override
    {
        if (ctx.nrows <= 0 || ctx.ncols <= 0)
            return 0;

        // Build columnsByRow for efficient candidate lookup.
        std::vector<std::vector<int>> columnsByRow(static_cast<size_t>(ctx.nrows));
        for (int j = 0; j < ctx.ncols; ++j)
        {
            if (!ctx.active[static_cast<size_t>(j)])
                continue;
            for (int row : ctx.rowsByColumn[static_cast<size_t>(j)])
                columnsByRow[static_cast<size_t>(row)].push_back(j);
        }

        // Sort columns by cost descending (most expensive first).
        std::vector<int> sortedCols;
        sortedCols.reserve(static_cast<size_t>(ctx.ncols));
        for (int j = 0; j < ctx.ncols; ++j)
        {
            if (ctx.active[static_cast<size_t>(j)])
                sortedCols.push_back(j);
        }
        std::sort(sortedCols.begin(), sortedCols.end(),
                  [&](int a, int b) { return ctx.costs[static_cast<size_t>(a)] > ctx.costs[static_cast<size_t>(b)]; });

        int removed = 0;
        std::vector<char> seen(static_cast<size_t>(ctx.ncols), 0);
        bool timedOut = false;

        for (int target : sortedCols)
        {
            if (std::chrono::steady_clock::now() >= ctx.deadline)
                break;
            if (!ctx.active[static_cast<size_t>(target)])
                continue;

            const std::vector<int> &targetRows = ctx.rowsByColumn[static_cast<size_t>(target)];
            if (targetRows.empty())
                continue;
            const double targetCost = ctx.costs[static_cast<size_t>(target)];

            // Collect candidate columns that share at least one row with target.
            std::vector<int> candidates;
            for (int row : targetRows)
            {
                for (int col : columnsByRow[static_cast<size_t>(row)])
                {
                    if (col != target && ctx.active[static_cast<size_t>(col)] && !seen[static_cast<size_t>(col)])
                    {
                        seen[static_cast<size_t>(col)] = 1;
                        candidates.push_back(col);
                    }
                }
            }
            // Reset seen for next target.
            for (int col : candidates)
                seen[static_cast<size_t>(col)] = 0;

            // Sort candidates by cost ascending for early termination.
            std::sort(candidates.begin(), candidates.end(),
                      [&](int a, int b) { return ctx.costs[static_cast<size_t>(a)] < ctx.costs[static_cast<size_t>(b)]; });

            bool dominated = false;

            // Try pairs.
            for (size_t i = 0; i < candidates.size() && !dominated; ++i)
            {
                if ((i & 255) == 0 && std::chrono::steady_clock::now() >= ctx.deadline)
                {
                    timedOut = true;
                    break;
                }
                const int a = candidates[i];
                const double costA = ctx.costs[static_cast<size_t>(a)];
                if (costA > targetCost + tol)
                    break;
                for (size_t j = i + 1; j < candidates.size() && !dominated; ++j)
                {
                    const int b = candidates[j];
                    if (costA + ctx.costs[static_cast<size_t>(b)] > targetCost + tol)
                        break;
                    if (unionCoversSorted(targetRows,
                                          ctx.rowsByColumn[static_cast<size_t>(a)],
                                          ctx.rowsByColumn[static_cast<size_t>(b)]))
                    {
                        dominated = true;
                    }
                }
            }

            if (timedOut)
                break;

            // Try triplets.
            if (!dominated)
            {
                for (size_t i = 0; i < candidates.size() && !dominated; ++i)
                {
                    if ((i & 63) == 0 && std::chrono::steady_clock::now() >= ctx.deadline)
                    {
                        timedOut = true;
                        break;
                    }
                    const int a = candidates[i];
                    const double costA = ctx.costs[static_cast<size_t>(a)];
                    if (costA > targetCost + tol)
                        break;
                    for (size_t j = i + 1; j < candidates.size() && !dominated; ++j)
                    {
                        const int b = candidates[j];
                        const double costAB = costA + ctx.costs[static_cast<size_t>(b)];
                        if (costAB > targetCost + tol)
                            break;
                        for (size_t k = j + 1; k < candidates.size() && !dominated; ++k)
                        {
                            const int c = candidates[k];
                            if (costAB + ctx.costs[static_cast<size_t>(c)] > targetCost + tol)
                                break;
                            if (tripleUnionCoversSorted(targetRows,
                                                        ctx.rowsByColumn[static_cast<size_t>(a)],
                                                        ctx.rowsByColumn[static_cast<size_t>(b)],
                                                        ctx.rowsByColumn[static_cast<size_t>(c)]))
                            {
                                dominated = true;
                            }
                        }
                    }
                }
            }

            if (timedOut)
                break;

            if (dominated)
            {
                ctx.active[static_cast<size_t>(target)] = 0;
                ++removed;
            }
        }
        return removed;
    }
};

} // namespace

std::vector<std::unique_ptr<IColumnPreprocessRule>> makeColumnPreprocessRules(const std::string &configured)
{
    std::vector<std::unique_ptr<IColumnPreprocessRule>> rules;
    const std::vector<std::string> tokens = splitCsvTokens(configured);
    if (tokens.empty())
    {
        rules.push_back(std::make_unique<SingleColumnDominanceRule>());
        rules.push_back(std::make_unique<TwoColumnDominanceRule>());
        return rules;
    }

    for (const std::string &token : tokens)
    {
        if (token == "none")
        {
            rules.clear();
            return rules;
        }
        if (token == "single_column_dominance" || token == "single")
        {
            rules.push_back(std::make_unique<SingleColumnDominanceRule>());
        }
        else if (token == "two_column_dominance" || token == "pair" || token == "two")
        {
            rules.push_back(std::make_unique<TwoColumnDominanceRule>());
        }
        else if (token == "cost_driven_replacement" || token == "cost_driven")
        {
            rules.push_back(std::make_unique<CostDrivenReplacementRule>());
        }
    }

    if (rules.empty())
    {
        rules.push_back(std::make_unique<SingleColumnDominanceRule>());
        rules.push_back(std::make_unique<TwoColumnDominanceRule>());
    }

    return rules;
}
