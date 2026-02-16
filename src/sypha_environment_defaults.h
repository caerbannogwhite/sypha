#ifndef SYPHA_ENVIRONMENT_DEFAULTS_H
#define SYPHA_ENVIRONMENT_DEFAULTS_H

#include "common.h"

namespace sypha_environment_defaults
{
constexpr double kPxInfinity = 1e50;
constexpr double kPxTolerance = 1e-12;

constexpr int kVerbosityLevel = 5;
constexpr int kCudaDeviceId = -1;

constexpr int kMehrotraMaxIter = 25;
constexpr double kMehrotraEta = 0.95;
constexpr double kMehrotraMuTol = 1e-4;
constexpr double kMehrotraCholTol = 1e-8;
constexpr int kMehrotraReorder = 1;
constexpr double kDenseGpuMemoryFractionThreshold = 2.0 / 3.0;

constexpr int kBnbMaxNodes = 100000;
constexpr int kBnbDeviceQueueCapacity = 1000;
constexpr int kBnbGapStallBranchIters = 5;
constexpr double kBnbGapStallMinImprovPct = 1.0;
constexpr double kBnbIntegralityTol = 1e-6;
constexpr int kBnbHeuristicEveryNNodes = 1;
constexpr double kBnbLogIntervalSeconds = 5.0;
constexpr double kBnbHardTimeLimitSeconds = 0.0;
constexpr bool kBnbDisable = false;
constexpr bool kBnbAutoFallbackLp = true;
constexpr bool kShowSolution = false;

inline const std::string &kBnbVarSelectionStrategy()
{
    static const std::string value = "most_fractional";
    return value;
}

inline const std::string &kBnbIntHeuristics()
{
    static const std::string value = "nearest_integer_fixing,dual_guided_cover_repair";
    return value;
}
} // namespace sypha_environment_defaults

#endif // SYPHA_ENVIRONMENT_DEFAULTS_H
