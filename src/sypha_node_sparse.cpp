
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

    //this->h_csrMatIndices = new std::vector<int>();
    //this->h_csrMatIndPtrs = new std::vector<int>();
    //this->h_csrMatVals = new std::vector<double>();
    
    this->h_cooMat = new std::vector<SyphaCOOEntry>();

    this->setInitValues();
    this->setUpCuda();
}

SyphaNodeSparse::~SyphaNodeSparse()
{

    cusparseDestroySpMat(this->matDescr);
    cusparseDestroyDnVec(this->objDescr);
    cusparseDestroyDnVec(this->rhsDescr);

    cudaFree(this->d_csrMatIndices);
    cudaFree(this->d_csrMatIndPtrs);
    cudaFree(this->d_csrMatVals);
    cudaFree(this->d_ObjDns);
    cudaFree(this->d_RhsDns);

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
    solver_sparse_merhrotra(*this);

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::importModel()
{
    if (this->env->modelType == MODEL_TYPE_SCP)
    {
        model_reader_read_scp_file_sparse_coo(*this, this->env->inputFilePath);
        //model_reader_read_scp_file_sparse_csr(*this, this->env->inputFilePath);
    } else {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNodeSparse::copyModelOnDevice()
{
    int nnz, k, prevRow;
    int *csrInds, *csrOffs;
    double *csrVals;

    nnz = this->h_cooMat->size();
    csrInds = (int *)malloc(nnz * sizeof(int));
    csrOffs = (int *)malloc((this->numRows + 1) * sizeof(int));
    csrVals = (double *)malloc(nnz * sizeof(double));

    k = 0;
    prevRow = -1;
    for (int i = 0; i < nnz; ++i)
    {
        csrInds[i] = this->h_cooMat->at(i).col;
        csrVals[i] = this->h_cooMat->at(i).val;

        if (this->h_cooMat->at(i).row != prevRow)
        {
            csrOffs[k] = i;
            prevRow = this->h_cooMat->at(i).row;
            k++;
        }
    }
    csrOffs[k] = nnz;

    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatIndices, sizeof(int) * nnz));
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatIndPtrs, sizeof(int) * (this->numRows + 1)));
    checkCudaErrors(cudaMalloc((void **)&this->d_csrMatVals,    sizeof(double) * nnz));
    checkCudaErrors(cudaMalloc((void **)&this->d_ObjDns,        sizeof(double) * this->numCols));
    checkCudaErrors(cudaMalloc((void **)&this->d_RhsDns,        sizeof(int) * this->numRows));

    checkCudaErrors(cudaMemcpyAsync(d_csrMatIndices,
                                    csrInds,
                                    sizeof(int) * nnz,
                                    cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(d_csrMatIndPtrs,
                                    csrOffs,
                                    sizeof(int) * (this->numRows + 1),
                                    cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(d_csrMatVals,
                                    csrVals,
                                    sizeof(int) * nnz,
                                    cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(d_ObjDns,
                                    h_ObjDns,
                                    sizeof(int) * this->numCols,
                                    cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(d_RhsDns,
                                    h_RhsDns,
                                    sizeof(int) * this->numRows,
                                    cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cusparseCreateCsr(&this->matDescr,
                                      this->numRows,
                                      this->numCols,
                                      this->numNonZero,
                                      this->d_csrMatIndPtrs,
                                      this->d_csrMatIndices,
                                      this->d_csrMatVals,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->objDescr,
                                        (int64_t)this->numCols,
                                        this->h_ObjDns,
                                        CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->rhsDescr,
                                        (int64_t)this->numRows,
                                        this->h_RhsDns,
                                        CUDA_R_64F));

    free(csrInds);
    free(csrOffs);
    free(csrVals);

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