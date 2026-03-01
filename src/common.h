#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <memory>
#include <limits>
#include <cmath>
#include <ctime>
#include <chrono>

#include "sypha_cuda_helper.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cusolverSp.h>

enum SyphaStatus
{
    CODE_SUCCESSFUL,
    CODE_GENERIC_ERROR,
    CODE_MODEL_TYPE_NOT_FOUND,
};

enum ModelInputType
{
    MODEL_TYPE_SCP = 0,
};

int utils_printDmat(int m, int n, int l, double *mat, bool device, bool trans);
int utils_printDvec(int n, double *vec, bool device);
int utils_printIvec(int n, int *vec, bool device);

#endif // COMMON_H