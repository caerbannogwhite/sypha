#ifndef SYPHA_NODE_SPARSE_H
#define SYPHA_NODE_SPARSE_H

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

#include "common.h"
#include "sypha_environment.h"
#include "sypha_cuda_helper.h"
#include "model_reader.h"

class SyphaEnvironment;

class SyphaNodeSparse
{
private:
    int numCols;
    int numRows;
    int numNonZero;
    double objectiveValue;
    int *h_csrMatIndices;
    int *h_csrMatIndPtrs;
    double *h_csrMatVals;
    double *h_ObjDns;
    double *h_RhsDns;
    int *d_csrMatIndices;
    int *d_csrMatIndPtrs;
    double *d_csrMatVals;
    double *d_ObjDns;
    double *d_RhsDns;

    
    SyphaEnvironment *env;

    cudaStream_t cudaStream;
    cusparseHandle_t cusparseHandle;
    cusolverSpHandle_t cusolverSpHandle;

public:
    SyphaNodeSparse(SyphaEnvironment &env);
    ~SyphaNodeSparse();

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

    friend SyphaStatus model_reader_read_scp_file_sparse(SyphaNodeSparse &node, string inputFilePath);
    friend SyphaStatus model_reader_scp_model_to_standard_sparse(SyphaNodeSparse &node);
};

#endif // SYPHA_NODE_SPARSE_H
