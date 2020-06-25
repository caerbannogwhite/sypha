#ifndef SYPHA_NODE_SPARSE_H
#define SYPHA_NODE_SPARSE_H

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

#include "common.h"
#include "sypha_environment.h"
#include "sypha_cuda_helper.h"

class SyphaEnvironment;

class SyphaCOOEntry
{
public:
    SyphaCOOEntry(int r, int c, double v);

    int row;
    int col;
    double val;
};

class SyphaNodeSparse
{
private:
    int ncols;
    int nrows;
    int nnz;
    double objvalPrim;
    double objvalDual;

    vector<SyphaCOOEntry> *h_cooMat;

    vector<int> *h_csrMatInds;
    vector<int> *h_csrMatOffs;
    vector<double> *h_csrMatVals;
    double *h_ObjDns;
    double *h_RhsDns;
    double *h_x;
    double *h_y;
    double *h_s;

    int *d_csrMatInds;
    int *d_csrMatOffs;
    double *d_csrMatVals;
    int *d_csrMatTransInds;
    int *d_csrMatTransOffs;
    double *d_csrMatTransVals;
    double *d_ObjDns;
    double *d_RhsDns;
    double *d_x;
    double *d_y;
    double *d_s;

    int iterations;
    double timeStartSolEnd;
    double timeStartSolStart;
    double timePreSolEnd;
    double timePreSolStart;
    double timeSolverEnd;
    double timeSolverStart;

    cusparseSpMatDescr_t matDescr;
    cusparseSpMatDescr_t matTransDescr;
    cusparseDnVecDescr_t objDescr;
    cusparseDnVecDescr_t rhsDescr;

    SyphaEnvironment *env;

    cudaStream_t cudaStream;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusolverDnHandle_t cusolverDnHandle;
    cusolverSpHandle_t cusolverSpHandle;

public:
    SyphaNodeSparse(SyphaEnvironment &env);
    ~SyphaNodeSparse();

    int getNumCols();
    int getNumRows();
    int getNumNonZero();
    int getIterations();
    double getObjvalPrim();
    double getObjvalDual();
    double getTimeStartSol();
    double getTimePreSol();
    double getTimeSolver();
    SyphaStatus solve();
    SyphaStatus readModel();
    SyphaStatus copyModelOnDevice();
    SyphaStatus setInitValues();
    SyphaStatus setUpCuda();

    friend SyphaStatus model_reader_read_scp_file_sparse_coo(SyphaNodeSparse &node, string inputFilePath);
    friend SyphaStatus model_reader_read_scp_file_sparse_csr(SyphaNodeSparse &node, string inputFilePath);
    friend SyphaStatus solver_sparse_mehrotra(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node);
};

#endif // SYPHA_NODE_SPARSE_H
