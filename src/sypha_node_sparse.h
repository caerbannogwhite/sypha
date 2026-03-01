#ifndef SYPHA_NODE_SPARSE_H
#define SYPHA_NODE_SPARSE_H

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverDn.h"
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
public:
    // Model dimensions
    int ncols;
    int nrows;
    int ncolsOriginal;      // Active original columns (before adding branch slacks)
    int ncolsInputOriginal; // Original columns from input instance before preprocessing
    int nnz;
    double objvalPrim;
    double objvalDual;
    double mipGap;

    // Host-side model data
    std::vector<SyphaCOOEntry> hCooMat;
    std::vector<int> hCsrMatInds;
    std::vector<int> hCsrMatOffs;
    std::vector<double> hCsrMatVals;
    std::vector<int> hActiveToInputCols; // active original col index -> input original col index
    std::vector<double> hObjDns;
    std::vector<double> hRhsDns;
    std::vector<double> hX;
    std::vector<double> hY;
    std::vector<double> hS;

    // Device-side model data
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

    // Solver statistics
    int iterations;
    double timeStartSolEnd;
    double timeStartSolStart;
    double timePreSolEnd;
    double timePreSolStart;
    double timeSolverEnd;
    double timeSolverStart;

    // cuSPARSE descriptors
    cusparseSpMatDescr_t matDescr;
    cusparseSpMatDescr_t matTransDescr;
    cusparseDnVecDescr_t objDescr;
    cusparseDnVecDescr_t rhsDescr;

    // Environment and CUDA handles
    SyphaEnvironment *env;
    cudaStream_t cudaStream;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusolverDnHandle_t cusolverDnHandle;
    cusolverSpHandle_t cusolverSpHandle;

    SyphaNodeSparse(SyphaEnvironment &env);
    ~SyphaNodeSparse();

    int getNumCols() const;
    int getNumRows() const;
    int getNumNonZero() const;
    int getIterations() const;
    double getObjvalPrim() const;
    double getObjvalDual() const;
    double getMipGap() const;
    double getTimeStartSol() const;
    double getTimePreSol() const;
    double getTimeSolver() const;
    SyphaStatus solve();
    SyphaStatus readModel();
    SyphaStatus preprocessModel(double incumbentUpperBound = std::numeric_limits<double>::infinity());
    SyphaStatus reduceByIncumbent(double incumbentUpperBound);
    SyphaStatus applyDominancePreprocessing();
    SyphaStatus applyCostDrivenReduction();
    SyphaStatus copyModelOnDevice();
    SyphaStatus setInitValues();
    SyphaStatus setUpCuda();
    SyphaStatus releaseModelOnDevice();

private:
    void initActiveColTracking();
    void rebuildCsrAfterRemoval(
        const std::vector<int> &oldToNew,
        const std::vector<int> &newToOld,
        const std::vector<int> &newActiveToInput,
        const double *oldCosts);
};

#endif // SYPHA_NODE_SPARSE_H
