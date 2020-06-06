
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
    return this->numCols;
}

int SyphaNodeDense::getNumRows()
{
    return this->numRows;
}

int SyphaNodeDense::getNumNonZero()
{
    return this->numRows;
}

double SyphaNodeDense::getObjectiveValue()
{
    return this->objectiveValue;
}

SyphaStatus SyphaNodeDense::solve()
{

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::importModel()
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
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::setInitValues()
{
    this->numCols = 0;
    this->numRows = 0;
    this->numNonZero = 0;

    this->objectiveValue = 0.0;

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeDense::setUpCuda()
{
    // initialize a cuda stream for this node
    checkCudaErrors(cudaStreamCreate(&this->cudaStream));

    // initialize cusolver Sparse, cusparse
    if (this->sparse)
    {
        checkCudaErrors(cusolverSpCreate(&this->cusolverSpHandle));
        checkCudaErrors(cusparseCreate(&this->cusparseHandle));

        // bind stream to cusparse and cusolver
        checkCudaErrors(cusolverSpSetStream(this->cusolverSpHandle, this->cudaStream));
        checkCudaErrors(cusparseSetStream(this->cusparseHandle, this->cudaStream));
    }

    // initialize cusolver Dense and cublas
    else
    {
        checkCudaErrors(cusolverDnCreate(&this->cusolverDnHandle));
        checkCudaErrors(cublasCreate(&this->cublasHandle));

        // bind stream to cublas and cusolver
        checkCudaErrors(cusolverDnSetStream(this->cusolverDnHandle, this->cudaStream));
        checkCudaErrors(cublasSetStream(this->cublasHandle, this->cudaStream));
    }

    return CODE_SUCCESFULL;
}