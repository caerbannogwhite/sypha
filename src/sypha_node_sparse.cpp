
#include "sypha_node_sparse.h"

SyphaNodeSparse::SyphaNodeSparse(SyphaEnvironment &env)
{
    this->env = &env;
    
    this->setInitValues();
    this->setUpCuda();
}

SyphaNodeSparse::~SyphaNodeSparse()
{
    checkCudaErrors(cusolverSpDestroy(this->cusolverSpHandle));
    checkCudaErrors(cusparseDestroy(this->cusparseHandle));
}

int SyphaNodeSparse::getNumCols()
{
    return this->numCols;
}

int SyphaNodeSparse::getNumRows()
{
    return this->numRows;
}

int SyphaNodeSparse::getNumNonZero()
{
    return this->numRows;
}

double SyphaNodeSparse::getObjectiveValue()
{
    return this->objectiveValue;
}

SyphaStatus SyphaNodeSparse::solve()
{

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::importModel()
{
    if (this->env->modelType == MODEL_TYPE_SCP)
    {
        model_reader_read_scp_file_sparse(*this, this->env->inputFilePath);
        model_reader_scp_model_to_standard_sparse(*this);
    } else {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::copyModelOnDevice()
{
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::setInitValues()
{
    this->numCols = 0;
    this->numRows = 0;
    this->numNonZero = 0;

    this->objectiveValue = 0.0;

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::setUpCuda()
{
    // initialize a cuda stream for this node
    checkCudaErrors(cudaStreamCreate(&this->cudaStream));

    checkCudaErrors(cusolverSpCreate(&this->cusolverSpHandle));
    checkCudaErrors(cusparseCreate(&this->cusparseHandle));

    // bind stream to cusparse and cusolver
    checkCudaErrors(cusolverSpSetStream(this->cusolverSpHandle, this->cudaStream));
    checkCudaErrors(cusparseSetStream(this->cusparseHandle, this->cudaStream));

    return CODE_SUCCESFULL;
}