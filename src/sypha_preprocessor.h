#ifndef SYPHA_PREPROCESSOR_H
#define SYPHA_PREPROCESSOR_H

#include <limits>
#include <memory>
#include <string>
#include <vector>

struct GreedySetCoverResult
{
    bool feasible = false;
    double objective = std::numeric_limits<double>::infinity();
    std::vector<int> selectedColumns; // indices into current active column space
};

GreedySetCoverResult greedy_set_cover_heuristic(
    int nrows,
    int ncolsOriginal,
    const std::vector<int> &csrInds,
    const std::vector<int> &csrOffs,
    const std::vector<double> &csrVals,
    const double *objDns);

struct ColumnPreprocessContext
{
    int nrows = 0;
    int ncols = 0;
    std::vector<std::vector<int>> rowsByColumn;
    std::vector<double> costs;
    std::vector<char> active;
};

class IColumnPreprocessRule
{
public:
    virtual ~IColumnPreprocessRule() {}
    virtual const char *name() const = 0;
    virtual int apply(ColumnPreprocessContext &ctx, double tol) const = 0;
};

std::vector<std::unique_ptr<IColumnPreprocessRule>> makeColumnPreprocessRules(const std::string &configured);

#endif // SYPHA_PREPROCESSOR_H
