
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
    this->nnz = 0;
    this->objval = 0.0;

    this->h_cooMat = new std::vector<SyphaCOOEntry>();

    this->h_csrMatInds = new std::vector<int>();
    this->h_csrMatOffs = new std::vector<int>();
    this->h_csrMatVals = new std::vector<double>();
    
    h_ObjDns = NULL;
    h_RhsDns = NULL;
    h_x = NULL;
    h_y = NULL;
    h_s = NULL;

    d_csrMatInds = NULL;
    d_csrMatOffs = NULL;
    d_csrMatVals = NULL;

    d_csrMatTransInds = NULL;
    d_csrMatTransOffs = NULL;
    d_csrMatTransVals = NULL;

    d_ObjDns = NULL;
    d_RhsDns = NULL;
    d_x = NULL;
    d_y = NULL;
    d_s = NULL;

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
    if (this->h_cooMat) this->h_cooMat->clear();

    if (this->h_csrMatInds) this->h_csrMatInds->clear();
    if (this->h_csrMatOffs) this->h_csrMatOffs->clear();
    if (this->h_csrMatVals) this->h_csrMatVals->clear();

    if (this->h_ObjDns) free(this->h_ObjDns);
    if (this->h_RhsDns) free(this->h_RhsDns);
    if (this->h_x) free(h_x);
    if (this->h_y) free(h_y);
    if (this->h_s) free(h_s);

    if (this->d_csrMatInds) checkCudaErrors(cudaFree(this->d_csrMatInds));
    if (this->d_csrMatOffs) checkCudaErrors(cudaFree(this->d_csrMatOffs));
    if (this->d_csrMatVals) checkCudaErrors(cudaFree(this->d_csrMatVals));

    if (this->d_csrMatTransInds) checkCudaErrors(cudaFree(this->d_csrMatTransInds));
    if (this->d_csrMatTransOffs) checkCudaErrors(cudaFree(this->d_csrMatTransOffs));
    if (this->d_csrMatTransVals) checkCudaErrors(cudaFree(this->d_csrMatTransVals));

    if (this->d_ObjDns) checkCudaErrors(cudaFree(this->d_ObjDns));
    if (this->d_RhsDns) checkCudaErrors(cudaFree(this->d_RhsDns));
    if (this->d_x) checkCudaErrors(cudaFree(this->d_x));
    if (this->d_y) checkCudaErrors(cudaFree(this->d_y));
    if (this->d_s) checkCudaErrors(cudaFree(this->d_s));

    if (this->matDescr) checkCudaErrors(cusparseDestroySpMat(this->matDescr));
    if (this->objDescr) checkCudaErrors(cusparseDestroyDnVec(this->objDescr));
    if (this->rhsDescr) checkCudaErrors(cusparseDestroyDnVec(this->rhsDescr));

    if (this->cublasHandle) checkCudaErrors(cublasDestroy(this->cublasHandle));
    if (this->cusparseHandle) checkCudaErrors(cusparseDestroy(this->cusparseHandle));
    if (this->cusolverDnHandle) checkCudaErrors(cusolverDnDestroy(this->cusolverDnHandle));
    if (this->cusolverSpHandle) checkCudaErrors(cusolverSpDestroy(this->cusolverSpHandle));

    if (this->cudaStream) checkCudaErrors(cudaStreamDestroy(this->cudaStream));
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