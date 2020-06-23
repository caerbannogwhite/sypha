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

    int DEBUG_LEVEL;
    double PX_INFINITY;
    double PX_TOLERANCE;

    // Merhrotra parameters
    int MEHROTRA_MAX_ITER = 1000;
    double MEHROTRA_ETA = 0.9;
    double MEHROTRA_MU_TOL = 1.E-8;
    double MEHROTRA_CHOL_TOL = 1.E-8;

public:
    SyphaEnvironment();
    SyphaEnvironment(int argc, char *argv[]);

    int getVerbosityLevel();
    std::string getTest();
    SyphaStatus getStatus();

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
    friend SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node);

    ///////////////////             UNIT TESTS
    friend int test_launcher(SyphaEnvironment &env);
    friend int sypha_test_scp4(SyphaEnvironment &env, int &pass);
    friend int sypha_test_scp5(SyphaEnvironment &env, int &pass);
};

#endif // SYPHA_ENVIRONMENT_H