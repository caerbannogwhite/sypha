#ifndef SYPHA_NODE_H
#define SYPHA_NODE_H

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cusparse.h"
#include "cusolverSp.h"

#include "common.h"
#include "sypha_environment.h"
#include "sypha_cuda_helper.h"

class SyphaEnvironment;

class SyphaNode
{
private:
    bool sparse;
    int numCols;
    int numRows;
    int numNonZero;
    double objectiveValue;
    double *h_MatDns;
    double *h_ObjDns;
    double *h_RhsDns;
    double *d_MatDns;
    double *d_ObjDns;
    double *d_RhsDns;

    
    SyphaEnvironment *env;

    cudaStream_t cudaStream;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverDnHandle;
    cusparseHandle_t cusparseHandle;
    cusolverSpHandle_t cusolverSpHandle;

public:
    SyphaNode(SyphaEnvironment &env);

    bool isSparse();
    int getNumCols();
    int getNumRows();
    int getNumNonZero();
    double getObjectiveValue();
    SyphaStatus solve();
    SyphaStatus importModel();
    SyphaStatus copyModelOnDevice();
    SyphaStatus convert2MySimplexForm();
    SyphaStatus setInitValues();
    SyphaStatus setUpCuda();

    friend SyphaStatus model_reader_read_scp_file_dense(SyphaNode &node, string inputFilePath);
    friend SyphaStatus model_reader_read_scp_file_sparse(SyphaNode &node, string inputFilePath);
    friend SyphaStatus model_reader_scp_model_to_standard_dense(SyphaNode &node);
    friend SyphaStatus model_reader_scp_model_to_standard_sparse(SyphaNode &node);
};

#endif // SYPHA_NODE_H
