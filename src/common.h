#ifndef COMMON_H
#define COMMON_H

#include <boost/program_options.hpp>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>

#include "sypha_cuda_helper.h"
#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

using namespace std;
namespace po = boost::program_options;

enum SyphaStatus
{
    CODE_SUCCESFULL,
    CODE_ERROR,
    CODE_MODEL_TYPE_NOT_FOUND,
};

enum ModelInputType
{
    MODEL_TYPE_SCP = 0,
};

int utils_printDmat(int m, int n, int l, double *mat, bool device);
int utils_printImat(int m, int n, int l, int *mat, bool device);
int utils_printDvec(int n, double *vec, bool device);
int utils_printIvec(int n, int *vec, bool device);

#endif // COMMON_H