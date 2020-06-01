#ifndef COMMON_H
#define COMMON_H

#include <boost/program_options.hpp>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>

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

#endif // COMMON_H