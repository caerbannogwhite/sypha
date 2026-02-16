#include "sypha_preprocessor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
                   { return (char)std::tolower(c); });
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
            if (!ctx.active[(size_t)target])
            {
                continue;
            }
            const std::vector<int> &targetRows = ctx.rowsByColumn[(size_t)target];
            for (int cand = 0; cand < ctx.ncols; ++cand)
            {
                if (cand == target || !ctx.active[(size_t)cand])
                {
                    continue;
                }
                if (ctx.costs[(size_t)cand] > ctx.costs[(size_t)target] + tol)
                {
                    continue;
                }

                const std::vector<int> &candRows = ctx.rowsByColumn[(size_t)cand];
                if (!isSubsetSorted(targetRows, candRows))
                {
                    continue;
                }

                if (fabs(ctx.costs[(size_t)cand] - ctx.costs[(size_t)target]) <= tol && cand > target)
                {
                    continue;
                }

                ctx.active[(size_t)target] = 0;
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
        for (int target = 0; target < ctx.ncols; ++target)
        {
            if (!ctx.active[(size_t)target])
            {
                continue;
            }

            const std::vector<int> &targetRows = ctx.rowsByColumn[(size_t)target];
            const double targetCost = ctx.costs[(size_t)target];
            bool dominated = false;

            for (int a = 0; a < ctx.ncols && !dominated; ++a)
            {
                if (a == target || !ctx.active[(size_t)a])
                {
                    continue;
                }
                const double aCost = ctx.costs[(size_t)a];
                if (aCost >= targetCost - tol)
                {
                    continue;
                }

                for (int b = a + 1; b < ctx.ncols; ++b)
                {
                    if (b == target || !ctx.active[(size_t)b])
                    {
                        continue;
                    }
                    const double pairCost = aCost + ctx.costs[(size_t)b];
                    if (pairCost >= targetCost - tol)
                    {
                        continue;
                    }
                    if (unionCoversSorted(targetRows, ctx.rowsByColumn[(size_t)a], ctx.rowsByColumn[(size_t)b]))
                    {
                        dominated = true;
                        break;
                    }
                }
            }

            if (dominated)
            {
                ctx.active[(size_t)target] = 0;
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
        rules.push_back(std::unique_ptr<IColumnPreprocessRule>(new SingleColumnDominanceRule()));
        rules.push_back(std::unique_ptr<IColumnPreprocessRule>(new TwoColumnDominanceRule()));
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
            rules.push_back(std::unique_ptr<IColumnPreprocessRule>(new SingleColumnDominanceRule()));
        }
        else if (token == "two_column_dominance" || token == "pair" || token == "two")
        {
            rules.push_back(std::unique_ptr<IColumnPreprocessRule>(new TwoColumnDominanceRule()));
        }
    }

    if (rules.empty())
    {
        rules.push_back(std::unique_ptr<IColumnPreprocessRule>(new SingleColumnDominanceRule()));
        rules.push_back(std::unique_ptr<IColumnPreprocessRule>(new TwoColumnDominanceRule()));
    }

    return rules;
}
