#pragma once

#include <boost/program_options.hpp>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>

using namespace std;
namespace po = boost::program_options;


typedef struct Instance
{
    int ncols;
    int nrows;
    double* hostMat;
    double* hostObj;
    double* hostRhs;
    double* localMat;
    double* localObj;
    double* localRhs;
    double* deviceMat;
    double* deviceObj;
    double* deviceRhs;
} Instance;


typedef struct Parameters
{
    // params
    string inputFilePath;
    int seed;
    int threadNum;
    double timeLimit;

    int DEBUG_LEVEL;
    double PX_INFINITY;
    double PX_TOLERANCE;
} Parameters;


enum SyphaStatus
{
    CODE_SUCCESFULL = 0,
    CODE_ERROR = 1
};


double comm_get_time_sec();
SyphaStatus comm_my_simplex_form(Instance &inst);
SyphaStatus comm_parse_program_args(Parameters &params, int argc, char* argv[]);
SyphaStatus comm_read_input_file(Instance &inst, Parameters &params);
SyphaStatus comm_free_instance(Instance &inst);
