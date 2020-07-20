#ifndef SYPHA_TEST_H
#define SYPHA_TEST_H

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

#include "common.h"
#include "sypha_environment.h"
#include "sypha_cuda_helper.h"

int test_launcher(SyphaEnvironment &env);

int sypha_test_001();

int sypha_test_scp4(SyphaEnvironment &env, int &pass);
int sypha_test_scp5(SyphaEnvironment &env, int &pass);

#endif // SYPHA_TEST_H