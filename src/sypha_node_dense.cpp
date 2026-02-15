
#include "sypha_node_dense.h"

SyphaNodeDense::SyphaNodeDense(SyphaEnvironment &env)
{
    this->env = &env;
    
    this->setInitValues();
    this->setUpCuda();
}

SyphaNodeDense::~SyphaNodeDense()
{
    checkCudaErrors(cusolverDnDestroy(this->cusolverDnHandle));
    checkCudaErrors(cublasDestroy(this->cublasHandle));
}

int SyphaNodeDense::getNumCols()
{
    return this->ncols;
}

int SyphaNodeDense::getNumRows()
{
    return this->nrows;
}

int SyphaNodeDense::getNumNonZero()
{
    return this->nnz;
}

double SyphaNodeDense::getObjval()
{
    return this->objval;
}

SyphaStatus SyphaNodeDense::solve()
{
    solver_dense_mehrotra(*this);

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::readModel()
{
    if (this->env->modelType == MODEL_TYPE_SCP)
    {
        model_reader_read_scp_file_dense(*this, this->env->inputFilePath);
    } else {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::copyModelOnDevice()
{
    checkCudaErrors(cudaMalloc((void **)&this->dMatDns, sizeof(double) * this->nrows * this->ncols));
    checkCudaErrors(cudaMalloc((void **)&this->dObjDns, sizeof(double) * this->ncols));
    checkCudaErrors(cudaMalloc((void **)&this->dRhsDns, sizeof(double) * this->nrows));

    checkCudaErrors(cudaMemcpyAsync(this->dMatDns, this->hMatDns,
                                    sizeof(double) * this->nrows * this->ncols, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dObjDns, this->hObjDns,
                                    sizeof(double) * this->ncols, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dRhsDns, this->hRhsDns,
                                    sizeof(double) * this->nrows, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::setInitValues()
{
    this->ncols = 0;
    this->nrows = 0;
    this->ncolsOriginal = 0;
    this->nnz = 0;

    this->objval = 0.0;

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::setUpCuda()
{
    // initialize a cuda stream for this node
    checkCudaErrors(cudaStreamCreate(&this->cudaStream));

    checkCudaErrors(cusolverDnCreate(&this->cusolverDnHandle));
    checkCudaErrors(cublasCreate(&this->cublasHandle));

    // bind stream to cublas and cusolver
    checkCudaErrors(cusolverDnSetStream(this->cusolverDnHandle, this->cudaStream));
    checkCudaErrors(cublasSetStream(this->cublasHandle, this->cudaStream));

    return CODE_SUCCESFULL;
}