
#include "sypha_node_sparse.h"

SyphaNodeSparse::SyphaNodeSparse(SyphaEnvironment &env)
{
    this->env = &env;

    this->h_csrMatIndices = new std::vector<int>();
    this->h_csrMatIndPtrs = new std::vector<int>();
    this->h_csrMatVals = new std::vector<double>();
    this->h_ObjDns = new std::vector<double>();
    this->h_RhsDns = new std::vector<double>();
    
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
    } else {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::copyModelOnDevice()
{
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatIndices, sizeof(int) * this->h_csrMatIndices->size()));
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatIndPtrs, sizeof(int) * this->h_csrMatIndPtrs->size()));
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatVals,    sizeof(double) * this->h_csrMatVals->size()));
    checkCudaErrors(cudaMalloc((void **)&this->d_ObjDns,        sizeof(double) * this->h_ObjDns->size()));
    checkCudaErrors(cudaMalloc((void **)&this->d_RhsDns,        sizeof(int) * this->h_RhsDns->size()));

    checkCudaErrors(cudaMemcpyAsync(d_csrMatIndices,    h_csrMatIndices->data(),    sizeof(int) * this->h_csrMatIndices->size(), cudaMemcpyHostToDevice, this->cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_csrMatIndPtrs,    h_csrMatIndPtrs->data(),    sizeof(int) * this->h_csrMatIndPtrs->size(), cudaMemcpyHostToDevice, this->cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_csrMatVals,       h_csrMatVals->data(),       sizeof(int) * this->h_csrMatVals->size(), cudaMemcpyHostToDevice, this->cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_ObjDns,           h_ObjDns->data(),           sizeof(int) * this->h_ObjDns->size(), cudaMemcpyHostToDevice, this->cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_RhsDns,           h_RhsDns->data(),           sizeof(int) * this->h_RhsDns->size(), cudaMemcpyHostToDevice, this->cudaStream));

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