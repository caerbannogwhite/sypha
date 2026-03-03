#ifndef SYPHA_ENVIRONMENT_H
#define SYPHA_ENVIRONMENT_H

#include <cuda_runtime.h>

#include "common.h"
#include "sypha_logger.h"

namespace sypha { class SolverImpl; }

class SyphaEnvironment
{
private:
    SyphaStatus internalStatus;
    std::string test;
    int testRepeat;
    std::string inputFilePath;
    ModelInputType modelType;
    bool sparse;

    int seed;
    int threadNum;
    int cudaDeviceId;
    int verbosityLevel;

    double timeLimit;

    int debugLevel;
    double pxInfinity;
    double pxTolerance;

    // Merhrotra parameters
    int mehrotraMaxIter;
    double mehrotraEta;
    double mehrotraMuTol;
    double mehrotraCholTol;
    int mehrotraReorder; /* 0 = no reorder, 1 = fill-reducing reorder for sparse QR */
    double denseGpuMemoryFractionThreshold;

    // Krylov CG solver parameters
    std::string linearSolverStrategy; // "auto", "dense", "sparse_qr", "krylov"
    int krylovMaxCgIter;
    double krylovCgTolInitial;
    double krylovCgTolFinal;
    double krylovCgTolDecayRate;

    // Branch-and-bound parameters
    int bnbMaxNodes;
    int bnbDeviceQueueCapacity;
    int bnbGapStallBranchIters;
    double bnbGapStallMinImprovPct;
    double bnbIntegralityTol;
    std::string bnbVarSelectionStrategy;
    int bnbHeuristicEveryNNodes;
    std::string bnbIntHeuristics;
    double bnbLogIntervalSeconds;
    double bnbHardTimeLimitSeconds;
    int bnbGapStagnationWindow;
    bool bnbDisable;
    bool bnbAutoFallbackLp;
    bool bnbCutsEnabled;
    int bnbCutRoundsRoot;
    int bnbMaxCutsPerRound;
    bool bnbBalasBranchingEnabled;
    int bnbBalasMaxBranches;
    bool showSolution;
    std::string preprocessColumnStrategies;
    double preprocessTimeLimitSeconds;

    std::unique_ptr<SyphaLogger> logger_;

public:
    SyphaEnvironment();
    SyphaEnvironment(int argc, char *argv[]);
    ~SyphaEnvironment();

    // Existing getters (now const)
    int getVerbosityLevel() const;
    std::string getTest() const;
    SyphaStatus getStatus() const;
    bool getShowSolution() const;
    SyphaLogger *getLogger() const;
    double timer() const;

    // Config parameter getters
    ModelInputType getModelType() const;
    const std::string &getInputFilePath() const;
    double getPxInfinity() const;
    double getPxTolerance() const;

    int getMehrotraMaxIter() const;
    double getMehrotraEta() const;
    double getMehrotraMuTol() const;
    double getMehrotraCholTol() const;
    int getMehrotraReorder() const;
    double getDenseGpuMemoryFractionThreshold() const;

    const std::string &getLinearSolverStrategy() const;
    int getKrylovMaxCgIter() const;
    double getKrylovCgTolInitial() const;
    double getKrylovCgTolFinal() const;
    double getKrylovCgTolDecayRate() const;

    int getBnbMaxNodes() const;
    int getBnbDeviceQueueCapacity() const;
    int getBnbGapStallBranchIters() const;
    double getBnbGapStallMinImprovPct() const;
    double getBnbIntegralityTol() const;
    const std::string &getBnbVarSelectionStrategy() const;
    int getBnbHeuristicEveryNNodes() const;
    const std::string &getBnbIntHeuristics() const;
    double getBnbLogIntervalSeconds() const;
    double getBnbHardTimeLimitSeconds() const;
    int getBnbGapStagnationWindow() const;
    bool getBnbDisable() const;
    bool getBnbAutoFallbackLp() const;
    bool getBnbCutsEnabled() const;
    int getBnbCutRoundsRoot() const;
    int getBnbMaxCutsPerRound() const;
    bool getBnbBalasBranchingEnabled() const;
    int getBnbBalasMaxBranches() const;

    const std::string &getPreprocessColumnStrategies() const;
    double getPreprocessTimeLimitSeconds() const;

    SyphaStatus setDefaultParameters();
    SyphaStatus setUpDevice();
    SyphaStatus readInputArguments(int argc, char *argv[]);

    friend class sypha::SolverImpl;
};

#endif // SYPHA_ENVIRONMENT_H