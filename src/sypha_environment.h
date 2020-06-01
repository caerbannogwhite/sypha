#ifndef SYPHA_ENVIRONMENT_H
#define SYPHA_ENVIRONMENT_H

#include <cuda_runtime.h>

#include "common.h"
#include "sypha_node_dense.h"
#include "sypha_node_sparse.h"

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

public:
    SyphaEnvironment();
    SyphaEnvironment(int argc, char *argv[]);

    SyphaStatus setDefaultParameters();
    SyphaStatus setUpDevice();
    SyphaStatus readInputArguments(int argc, char *argv[]);

    void logger(string message, string type, int level);

    friend class SyphaNodeDense;
    friend class SyphaNodeSparse;
};

#endif // SYPHA_ENVIRONMENT_H