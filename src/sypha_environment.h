#ifndef SYPHA_ENVIRONMENT_H
#define SYPHA_ENVIRONMENT_H

#include <cuda_runtime.h>

#include "common.h"
#include "sypha_solver_dense.h"
#include "sypha_solver_sparse.h"
#include "sypha_node_dense.h"
#include "sypha_node_sparse.h"

class SyphaNodeDense;
class SyphaNodeSparse;

class SyphaEnvironment
{
private:
    SyphaStatus internalStatus;
    string test;
    int testRepeat;
    string inputFilePath;
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

    // Branch-and-bound parameters
    int bnbMaxNodes;
    int bnbDeviceQueueCapacity;
    int bnbGapStallBranchIters;
    double bnbGapStallMinImprovPct;
    double bnbIntegralityTol;
    string bnbVarSelectionStrategy;
    int bnbHeuristicEveryNNodes;
    string bnbIntHeuristics;
    double bnbLogIntervalSeconds;
    double bnbHardTimeLimitSeconds;
    bool bnbDisable;
    bool bnbAutoFallbackLp;
    bool showSolution;

public:
    SyphaEnvironment();
    SyphaEnvironment(int argc, char *argv[]);

    int getVerbosityLevel();
    std::string getTest();
    SyphaStatus getStatus();
    bool getShowSolution();

    double timer();

    SyphaStatus setDefaultParameters();
    SyphaStatus setUpDevice();
    SyphaStatus readInputArguments(int argc, char *argv[]);

    void logger(string message, string type, int level);

    friend class SyphaNodeDense;
    friend class SyphaNodeSparse;

    friend SyphaStatus solver_dense_mehrotra(SyphaNodeDense &node);
    friend SyphaStatus solver_dense_mehrotra_init(SyphaNodeDense &node);
    friend SyphaStatus solver_sparse_mehrotra(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_run(SyphaNodeSparse &node, const SolverExecutionConfig &config, SolverExecutionResult *result);
    friend SyphaStatus solver_sparse_mehrotra_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_branch_and_bound(SyphaNodeSparse &node);

    ///////////////////             UNIT TESTS
    friend int test_launcher(SyphaEnvironment &env);
    friend int sypha_test_scp4(SyphaEnvironment &env, int &pass);
    friend int sypha_test_scp5(SyphaEnvironment &env, int &pass);
};

#endif // SYPHA_ENVIRONMENT_H