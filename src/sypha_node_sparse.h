#ifndef SYPHA_NODE_SPARSE_H
#define SYPHA_NODE_SPARSE_H

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

#include "common.h"
#include "sypha_environment.h"
#include "sypha_cuda_helper.h"

class SyphaEnvironment;
struct SolverExecutionConfig;
struct SolverExecutionResult;

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
    int ncolsOriginal;  // Original columns (before adding slack variables)
    int nnz;
    double objvalPrim;
    double objvalDual;
    double mipGap;

    vector<SyphaCOOEntry> *hCooMat;

    vector<int> *hCsrMatInds;
    vector<int> *hCsrMatOffs;
    vector<double> *hCsrMatVals;
    double *hObjDns;
    double *hRhsDns;
    double *hX;
    double *hY;
    double *hS;

    int *dCsrMatInds;
    int *dCsrMatOffs;
    double *dCsrMatVals;
    int *dCsrMatTransInds;
    int *dCsrMatTransOffs;
    double *dCsrMatTransVals;
    double *dObjDns;
    double *dRhsDns;
    double *dX;
    double *dY;
    double *dS;

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

    SyphaStatus releaseModelOnDevice();

public:
    SyphaNodeSparse(SyphaEnvironment &env);
    ~SyphaNodeSparse();

    int getNumCols();
    int getNumRows();
    int getNumNonZero();
    int getIterations();
    double getObjvalPrim();
    double getObjvalDual();
    double getMipGap();
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
    friend SyphaStatus solver_sparse_mehrotra_run(SyphaNodeSparse &node, const SolverExecutionConfig &config, SolverExecutionResult *result);
    friend SyphaStatus solver_sparse_mehrotra_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node);
    friend SyphaStatus solver_sparse_branch_and_bound(SyphaNodeSparse &node);
};

#endif // SYPHA_NODE_SPARSE_H
