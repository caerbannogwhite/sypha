
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

    this->h_csrMatInds = new std::vector<int>();
    this->h_csrMatOffs = new std::vector<int>();
    this->h_csrMatVals = new std::vector<double>();
    
    //this->h_cooMat = new std::vector<SyphaCOOEntry>();

    this->setInitValues();
    this->setUpCuda();
}

SyphaNodeSparse::~SyphaNodeSparse()
{

    cusparseDestroySpMat(this->matDescr);
    cusparseDestroyDnVec(this->objDescr);
    cusparseDestroyDnVec(this->rhsDescr);

    cudaFree(this->d_csrMatInds);
    cudaFree(this->d_csrMatOffs);
    cudaFree(this->d_csrMatVals);
    cudaFree(this->d_ObjDns);
    cudaFree(this->d_RhsDns);

    checkCudaErrors(cusolverSpDestroy(this->cusolverSpHandle));
    checkCudaErrors(cusparseDestroy(this->cusparseHandle));
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

double SyphaNodeSparse::getObjval()
{
    return this->objval;
}

SyphaStatus SyphaNodeSparse::solve()
{
    solver_sparse_merhrotra(*this);

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::readModel()
{
    if (this->env->modelType == MODEL_TYPE_SCP)
    {
        //model_reader_read_scp_file_sparse_coo(*this, this->env->inputFilePath);
        model_reader_read_scp_file_sparse_csr(*this, this->env->inputFilePath);
    } else {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::copyModelOnDevice()
{
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatInds, sizeof(int) * this->nnz));
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatOffs, sizeof(int) * (this->nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatVals, sizeof(double) * this->nnz));
    checkCudaErrors(cudaMalloc((void **)&this->d_ObjDns,     sizeof(double) * this->ncols));
    checkCudaErrors(cudaMalloc((void **)&this->d_RhsDns,     sizeof(double) * this->nrows));

    checkCudaErrors(cudaMemcpyAsync(this->d_csrMatInds, this->h_csrMatInds->data(),
                                    sizeof(int) * this->nnz, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->d_csrMatOffs, this->h_csrMatOffs->data(),
                                    sizeof(int) * (this->nrows + 1), cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->d_csrMatVals, this->h_csrMatVals->data(),
                                    sizeof(double) * this->nnz, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->d_ObjDns, this->h_ObjDns,
                                    sizeof(double) * this->ncols, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->d_RhsDns, this->h_RhsDns,
                                    sizeof(double) * this->nrows, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    // checkCudaErrors(cudaMemcpy(this->d_csrMatInds, this->h_csrMatInds->data(),
    //                            sizeof(int) * this->h_csrMatInds->size(), cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpy(this->d_csrMatOffs, this->h_csrMatOffs->data(),
    //                            sizeof(int) * (this->nrows + 1), cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpy(this->d_csrMatVals, this->h_csrMatVals->data(),
    //                            sizeof(double) * this->h_csrMatVals->size(), cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpy(this->d_ObjDns, this->h_ObjDns,
    //                            sizeof(double) * this->ncols, cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpy(this->d_RhsDns, this->h_RhsDns,
    //                            sizeof(double) * this->nrows, cudaMemcpyHostToDevice));

    checkCudaErrors(cusparseCreateCsr(&this->matDescr, this->nrows, this->ncols, this->nnz,
                                      this->d_csrMatOffs, this->d_csrMatInds, this->d_csrMatVals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->objDescr, (int64_t)this->ncols,
                                        this->h_ObjDns, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->rhsDescr, (int64_t)this->nrows,
                                        this->h_RhsDns, CUDA_R_64F));

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::setInitValues()
{
    this->ncols = 0;
    this->nrows = 0;
    this->nnz = 0;

    this->objval = 0.0;

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