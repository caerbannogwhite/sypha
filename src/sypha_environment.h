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
    int MERHROTRA_MAX_ITER;
    double MERHROTRA_MU_TOL;

public:
    SyphaEnvironment();
    SyphaEnvironment(int argc, char *argv[]);

    SyphaStatus setDefaultParameters();
    SyphaStatus setUpDevice();
    SyphaStatus readInputArguments(int argc, char *argv[]);

    void logger(string message, string type, int level);

    friend class SyphaNodeDense;
    friend class SyphaNodeSparse;

    friend SyphaStatus solver_dense_merhrotra(SyphaNodeDense &node);
    friend SyphaStatus solver_dense_merhrotra_init(SyphaNodeDense &node);
    friend SyphaStatus solver_sparse_merhrotra(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_merhrotra_init(SyphaNodeSparse &node);
};

#endif // SYPHA_ENVIRONMENT_H