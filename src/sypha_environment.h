#ifndef SYPHA_ENVIRONMENT_H
#define SYPHA_ENVIRONMENT_H

#include "common.h"
#include "sypha_node.h"

class SyphaEnvironment
{
private:
    string inputFilePath;
    ModelInputType modelType;
    bool sparse;
    int seed;
    int threadNum;
    double timeLimit;

    int DEBUG_LEVEL;
    double PX_INFINITY;
    double PX_TOLERANCE;

public:
    SyphaEnvironment();
    SyphaEnvironment(int argc, char *argv[]);

    bool getSparse();
    int getSeed();
    int getThreadNum();
    double getTimeLimit();

    SyphaStatus readInputArguments(int argc, char *argv[]);

    friend class SyphaNode;
};

#endif // SYPHA_ENVIRONMENT_H