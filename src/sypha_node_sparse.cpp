
#include "sypha_node_sparse.h"

SyphaCOOEntry::SyphaCOOEntry(int r, int c, double v)
{
    this->row = r;
    this->col = c;
    this->val = v;
}

SyphaNodeSparse::SyphaNodeSparse(SyphaEnvironment &env)
{
    this->env = &env;

    this->ncols = 0;
    this->nrows = 0;
    this->ncolsOriginal = 0;
    this->nnz = 0;
    this->objvalPrim = 0.0;
    this->objvalDual = 0.0;
    this->mipGap = std::numeric_limits<double>::infinity();

    this->hCooMat = new std::vector<SyphaCOOEntry>();

    this->hCsrMatInds = new std::vector<int>();
    this->hCsrMatOffs = new std::vector<int>();
    this->hCsrMatVals = new std::vector<double>();

    hObjDns = NULL;
    hRhsDns = NULL;
    hX = NULL;
    hY = NULL;
    hS = NULL;

    dCsrMatInds = NULL;
    dCsrMatOffs = NULL;
    dCsrMatVals = NULL;

    dCsrMatTransInds = NULL;
    dCsrMatTransOffs = NULL;
    dCsrMatTransVals = NULL;

    dObjDns = NULL;
    dRhsDns = NULL;
    dX = NULL;
    dY = NULL;
    dS = NULL;

    matDescr = NULL;
    matTransDescr = NULL;
    objDescr = NULL;
    rhsDescr = NULL;

    cudaStream = NULL;

    cublasHandle = NULL;
    cusparseHandle = NULL;
    cusolverDnHandle = NULL;
    cusolverSpHandle = NULL;

    this->setInitValues();
    this->setUpCuda();
}

SyphaNodeSparse::~SyphaNodeSparse()
{
    if (this->hCooMat)
        this->hCooMat->clear();

    if (this->hCsrMatInds)
        this->hCsrMatInds->clear();
    if (this->hCsrMatOffs)
        this->hCsrMatOffs->clear();
    if (this->hCsrMatVals)
        this->hCsrMatVals->clear();

    if (this->hObjDns)
        free(this->hObjDns);
    if (this->hRhsDns)
        free(this->hRhsDns);
    if (this->hX)
        free(hX);
    if (this->hY)
        free(hY);
    if (this->hS)
        free(hS);

    this->releaseModelOnDevice();

    if (this->dX)
        checkCudaErrors(cudaFree(this->dX));
    if (this->dY)
        checkCudaErrors(cudaFree(this->dY));
    if (this->dS)
        checkCudaErrors(cudaFree(this->dS));

    if (this->cublasHandle)
        checkCudaErrors(cublasDestroy(this->cublasHandle));
    if (this->cusparseHandle)
        checkCudaErrors(cusparseDestroy(this->cusparseHandle));
    if (this->cusolverDnHandle)
        checkCudaErrors(cusolverDnDestroy(this->cusolverDnHandle));
    if (this->cusolverSpHandle)
        checkCudaErrors(cusolverSpDestroy(this->cusolverSpHandle));

    if (this->cudaStream)
        checkCudaErrors(cudaStreamDestroy(this->cudaStream));
}

int SyphaNodeSparse::getNumCols()
{
    return this->ncols;
}

int SyphaNodeSparse::getNumRows()
{
    return this->nrows;
}

int SyphaNodeSparse::getNumNonZero()
{
    return this->nnz;
}

int SyphaNodeSparse::getIterations()
{
    return this->iterations;
}

double SyphaNodeSparse::getObjvalPrim()
{
    return this->objvalPrim;
}

double SyphaNodeSparse::getObjvalDual()
{
    return this->objvalDual;
}

double SyphaNodeSparse::getMipGap()
{
    return this->mipGap;
}

double SyphaNodeSparse::getTimeStartSol()
{
    return this->timeStartSolEnd - this->timeStartSolStart;
}

double SyphaNodeSparse::getTimePreSol()
{
    return this->timePreSolEnd - this->timePreSolStart;
}

double SyphaNodeSparse::getTimeSolver()
{
    return this->timeSolverEnd - this->timeSolverStart;
}

SyphaStatus SyphaNodeSparse::solve()
{
    if (this->env->bnbDisable)
    {
        return solver_sparse_mehrotra(*this);
    }
    return solver_sparse_branch_and_bound(*this);
}

SyphaStatus SyphaNodeSparse::readModel()
{
    if (this->env->modelType == MODEL_TYPE_SCP)
    {
        // model_reader_read_scp_file_sparse_coo(*this, this->env->inputFilePath);
        return model_reader_read_scp_file_sparse_csr(*this, this->env->inputFilePath);
    }
    else
    {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
}

SyphaStatus SyphaNodeSparse::copyModelOnDevice()
{
    this->releaseModelOnDevice();

    checkCudaErrors(cudaMalloc((void **)&this->dCsrMatInds, sizeof(int) * this->nnz));
    checkCudaErrors(cudaMalloc((void **)&this->dCsrMatOffs, sizeof(int) * (this->nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&this->dCsrMatVals, sizeof(double) * this->nnz));
    checkCudaErrors(cudaMalloc((void **)&this->dObjDns, sizeof(double) * this->ncols));
    checkCudaErrors(cudaMalloc((void **)&this->dRhsDns, sizeof(double) * this->nrows));

    checkCudaErrors(cudaMemcpyAsync(this->dCsrMatInds, this->hCsrMatInds->data(),
                                    sizeof(int) * this->nnz, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dCsrMatOffs, this->hCsrMatOffs->data(),
                                    sizeof(int) * (this->nrows + 1), cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dCsrMatVals, this->hCsrMatVals->data(),
                                    sizeof(double) * this->nnz, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dObjDns, this->hObjDns,
                                    sizeof(double) * this->ncols, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dRhsDns, this->hRhsDns,
                                    sizeof(double) * this->nrows, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cusparseCreateCsr(&this->matDescr, this->nrows, this->ncols, this->nnz,
                                      this->dCsrMatOffs, this->dCsrMatInds, this->dCsrMatVals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->objDescr, (int64_t)this->ncols,
                                        this->hObjDns, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->rhsDescr, (int64_t)this->nrows,
                                        this->hRhsDns, CUDA_R_64F));

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::releaseModelOnDevice()
{
    if (this->dCsrMatInds)
    {
        checkCudaErrors(cudaFree(this->dCsrMatInds));
        this->dCsrMatInds = NULL;
    }
    if (this->dCsrMatOffs)
    {
        checkCudaErrors(cudaFree(this->dCsrMatOffs));
        this->dCsrMatOffs = NULL;
    }
    if (this->dCsrMatVals)
    {
        checkCudaErrors(cudaFree(this->dCsrMatVals));
        this->dCsrMatVals = NULL;
    }

    if (this->dCsrMatTransInds)
    {
        checkCudaErrors(cudaFree(this->dCsrMatTransInds));
        this->dCsrMatTransInds = NULL;
    }
    if (this->dCsrMatTransOffs)
    {
        checkCudaErrors(cudaFree(this->dCsrMatTransOffs));
        this->dCsrMatTransOffs = NULL;
    }
    if (this->dCsrMatTransVals)
    {
        checkCudaErrors(cudaFree(this->dCsrMatTransVals));
        this->dCsrMatTransVals = NULL;
    }

    if (this->dObjDns)
    {
        checkCudaErrors(cudaFree(this->dObjDns));
        this->dObjDns = NULL;
    }
    if (this->dRhsDns)
    {
        checkCudaErrors(cudaFree(this->dRhsDns));
        this->dRhsDns = NULL;
    }

    if (this->matDescr)
    {
        checkCudaErrors(cusparseDestroySpMat(this->matDescr));
        this->matDescr = NULL;
    }
    if (this->matTransDescr)
    {
        checkCudaErrors(cusparseDestroySpMat(this->matTransDescr));
        this->matTransDescr = NULL;
    }
    if (this->objDescr)
    {
        checkCudaErrors(cusparseDestroyDnVec(this->objDescr));
        this->objDescr = NULL;
    }
    if (this->rhsDescr)
    {
        checkCudaErrors(cusparseDestroyDnVec(this->rhsDescr));
        this->rhsDescr = NULL;
    }

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::setInitValues()
{
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::setUpCuda()
{
    // initialize a cuda stream for this node
    checkCudaErrors(cudaStreamCreate(&this->cudaStream));

    checkCudaErrors(cublasCreate(&this->cublasHandle));
    checkCudaErrors(cusparseCreate(&this->cusparseHandle));
    checkCudaErrors(cusolverDnCreate(&this->cusolverDnHandle));
    checkCudaErrors(cusolverSpCreate(&this->cusolverSpHandle));

    // bind stream to cusparse and cusolver
    checkCudaErrors(cublasSetStream(this->cublasHandle, this->cudaStream));
    checkCudaErrors(cusparseSetStream(this->cusparseHandle, this->cudaStream));
    checkCudaErrors(cusolverDnSetStream(this->cusolverDnHandle, this->cudaStream));
    checkCudaErrors(cusolverSpSetStream(this->cusolverSpHandle, this->cudaStream));

    return CODE_SUCCESFULL;
}