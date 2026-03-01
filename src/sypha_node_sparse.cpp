
#include "sypha_node_sparse.h"
#include "sypha_solver_sparse.h"
#include "model_reader.h"
#include "sypha_preprocessor.h"

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
    this->ncolsInputOriginal = 0;
    this->nnz = 0;
    this->objvalPrim = 0.0;
    this->objvalDual = 0.0;
    this->mipGap = std::numeric_limits<double>::infinity();

    // Vectors are default-constructed (empty)

    dCsrMatInds = nullptr;
    dCsrMatOffs = nullptr;
    dCsrMatVals = nullptr;

    dCsrMatTransInds = nullptr;
    dCsrMatTransOffs = nullptr;
    dCsrMatTransVals = nullptr;

    dObjDns = nullptr;
    dRhsDns = nullptr;
    dX = nullptr;
    dY = nullptr;
    dS = nullptr;

    matDescr = nullptr;
    matTransDescr = nullptr;
    objDescr = nullptr;
    rhsDescr = nullptr;

    cudaStream = nullptr;

    cublasHandle = nullptr;
    cusparseHandle = nullptr;
    cusolverDnHandle = nullptr;
    cusolverSpHandle = nullptr;

    this->setInitValues();
    this->setUpCuda();
}

SyphaNodeSparse::~SyphaNodeSparse()
{
    // Vectors are auto-destructed

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

int SyphaNodeSparse::getNumCols() const
{
    return this->ncols;
}

int SyphaNodeSparse::getNumRows() const
{
    return this->nrows;
}

int SyphaNodeSparse::getNumNonZero() const
{
    return this->nnz;
}

int SyphaNodeSparse::getIterations() const
{
    return this->iterations;
}

double SyphaNodeSparse::getObjvalPrim() const
{
    return this->objvalPrim;
}

double SyphaNodeSparse::getObjvalDual() const
{
    return this->objvalDual;
}

double SyphaNodeSparse::getMipGap() const
{
    return this->mipGap;
}

double SyphaNodeSparse::getTimeStartSol() const
{
    return this->timeStartSolEnd - this->timeStartSolStart;
}

double SyphaNodeSparse::getTimePreSol() const
{
    return this->timePreSolEnd - this->timePreSolStart;
}

double SyphaNodeSparse::getTimeSolver() const
{
    return this->timeSolverEnd - this->timeSolverStart;
}

SyphaStatus SyphaNodeSparse::solve()
{
    if (this->env->getBnbDisable())
    {
        return solver_sparse_mehrotra(*this);
    }
    return solver_sparse_branch_and_bound(*this);
}

SyphaStatus SyphaNodeSparse::readModel()
{
    if (this->env->getModelType() == MODEL_TYPE_SCP)
    {
        return model_reader_read_scp_file_sparse_csr(*this, this->env->getInputFilePath());
    }
    else
    {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
}

SyphaStatus SyphaNodeSparse::copyModelOnDevice()
{
    this->releaseModelOnDevice();

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&this->dCsrMatInds), sizeof(int) * this->nnz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&this->dCsrMatOffs), sizeof(int) * (this->nrows + 1)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&this->dCsrMatVals), sizeof(double) * this->nnz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&this->dObjDns), sizeof(double) * this->ncols));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&this->dRhsDns), sizeof(double) * this->nrows));

    checkCudaErrors(cudaMemcpyAsync(this->dCsrMatInds, this->hCsrMatInds.data(),
                                    sizeof(int) * this->nnz, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dCsrMatOffs, this->hCsrMatOffs.data(),
                                    sizeof(int) * (this->nrows + 1), cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dCsrMatVals, this->hCsrMatVals.data(),
                                    sizeof(double) * this->nnz, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dObjDns, this->hObjDns.data(),
                                    sizeof(double) * this->ncols, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cudaMemcpyAsync(this->dRhsDns, this->hRhsDns.data(),
                                    sizeof(double) * this->nrows, cudaMemcpyHostToDevice,
                                    this->cudaStream));

    checkCudaErrors(cusparseCreateCsr(&this->matDescr, this->nrows, this->ncols, this->nnz,
                                      this->dCsrMatOffs, this->dCsrMatInds, this->dCsrMatVals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                      CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->objDescr, static_cast<int64_t>(this->ncols),
                                        this->hObjDns.data(), CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&this->rhsDescr, static_cast<int64_t>(this->nrows),
                                        this->hRhsDns.data(), CUDA_R_64F));

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaNodeSparse::preprocessModel(double incumbentUpperBound)
{
    SyphaStatus s = reduceByIncumbent(incumbentUpperBound);
    if (s != CODE_SUCCESSFUL)
        return s;
    return applyDominancePreprocessing();
}

void SyphaNodeSparse::initActiveColTracking()
{
    if (this->ncolsInputOriginal <= 0)
    {
        this->ncolsInputOriginal = this->ncolsOriginal;
    }
    if (this->hActiveToInputCols.empty())
    {
        this->hActiveToInputCols.resize(static_cast<size_t>(this->ncolsOriginal));
        for (int j = 0; j < this->ncolsOriginal; ++j)
        {
            this->hActiveToInputCols[static_cast<size_t>(j)] = j;
        }
    }
}

void SyphaNodeSparse::rebuildCsrAfterRemoval(
    const std::vector<int> &oldToNew,
    const std::vector<int> &newToOld,
    const std::vector<int> &newActiveToInput,
    const double *oldCosts)
{
    const int newOriginalCols = static_cast<int>(newActiveToInput.size());
    std::vector<int> newCsrInds;
    std::vector<int> newCsrOffs;
    std::vector<double> newCsrVals;
    std::vector<double> newObj(static_cast<size_t>(newOriginalCols + this->nrows), 0.0);

    newCsrOffs.reserve(static_cast<size_t>(this->nrows) + 1);
    newCsrOffs.push_back(0);

    for (int newCol = 0; newCol < newOriginalCols; ++newCol)
    {
        newObj[static_cast<size_t>(newCol)] = oldCosts[static_cast<size_t>(newToOld[static_cast<size_t>(newCol)])];
    }

    for (int i = 0; i < this->nrows; ++i)
    {
        const int begin = this->hCsrMatOffs[static_cast<size_t>(i)];
        const int end = this->hCsrMatOffs[static_cast<size_t>(i) + 1];
        for (int k = begin; k < end; ++k)
        {
            const int oldCol = this->hCsrMatInds[static_cast<size_t>(k)];
            const double val = this->hCsrMatVals[static_cast<size_t>(k)];
            if (oldCol >= 0 && oldCol < this->ncolsOriginal)
            {
                const int mapped = oldToNew[static_cast<size_t>(oldCol)];
                if (mapped >= 0)
                {
                    newCsrInds.push_back(mapped);
                    newCsrVals.push_back(val);
                }
                continue;
            }

            if (oldCol == this->ncolsOriginal + i)
            {
                newCsrInds.push_back(newOriginalCols + i);
                newCsrVals.push_back(val);
            }
        }
        newCsrOffs.push_back(static_cast<int>(newCsrVals.size()));
    }

    this->hObjDns.assign(newObj.begin(), newObj.end());

    this->hCsrMatInds = newCsrInds;
    this->hCsrMatOffs = newCsrOffs;
    this->hCsrMatVals = newCsrVals;
    this->hActiveToInputCols = newActiveToInput;

    this->ncolsOriginal = newOriginalCols;
    this->ncols = newOriginalCols + this->nrows;
    this->nnz = static_cast<int>(newCsrVals.size());
}

SyphaStatus SyphaNodeSparse::reduceByIncumbent(double incumbentUpperBound)
{
    if (this->nrows <= 0 || this->ncolsOriginal <= 0 || this->hObjDns.empty() ||
        this->hCsrMatInds.empty() || this->hCsrMatOffs.empty() || this->hCsrMatVals.empty())
    {
        return CODE_SUCCESSFUL;
    }
    if (!std::isfinite(incumbentUpperBound))
    {
        return CODE_SUCCESSFUL;
    }

    initActiveColTracking();

    std::vector<int> newActiveToInput;
    std::vector<int> newToOld;
    newActiveToInput.reserve(static_cast<size_t>(this->ncolsOriginal));
    newToOld.reserve(static_cast<size_t>(this->ncolsOriginal));
    std::vector<int> oldToNew(static_cast<size_t>(this->ncolsOriginal), -1);
    int removed = 0;

    for (int oldCol = 0; oldCol < this->ncolsOriginal; ++oldCol)
    {
        if (this->hObjDns[oldCol] + this->env->getPxTolerance() >= incumbentUpperBound)
        {
            ++removed;
            continue;
        }
        const int newCol = static_cast<int>(newActiveToInput.size());
        oldToNew[static_cast<size_t>(oldCol)] = newCol;
        newActiveToInput.push_back(this->hActiveToInputCols[static_cast<size_t>(oldCol)]);
        newToOld.push_back(oldCol);
    }

    if (removed == 0)
    {
        return CODE_SUCCESSFUL;
    }
    if (newActiveToInput.empty())
    {
        this->env->getLogger()->log(LOG_INFO, "Preprocessing removed all columns; keeping original model");
        return CODE_SUCCESSFUL;
    }

    std::vector<double> oldCosts(this->hObjDns.begin(), this->hObjDns.begin() + this->ncolsOriginal);
    rebuildCsrAfterRemoval(oldToNew, newToOld, newActiveToInput, oldCosts.data());

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaNodeSparse::applyDominancePreprocessing()
{
    if (this->nrows <= 0 || this->ncolsOriginal <= 0 || this->hObjDns.empty() ||
        this->hCsrMatInds.empty() || this->hCsrMatOffs.empty() || this->hCsrMatVals.empty())
    {
        return CODE_SUCCESSFUL;
    }

    initActiveColTracking();

    ColumnPreprocessContext ctx;
    ctx.nrows = this->nrows;
    ctx.ncols = this->ncolsOriginal;
    ctx.rowsByColumn.assign(static_cast<size_t>(ctx.ncols), std::vector<int>());
    ctx.costs.assign(this->hObjDns.begin(), this->hObjDns.begin() + this->ncolsOriginal);
    ctx.active.assign(static_cast<size_t>(ctx.ncols), 1);
    if (this->env->getPreprocessTimeLimitSeconds() > 0.0)
    {
        ctx.deadline = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(this->env->getPreprocessTimeLimitSeconds()));
    }

    for (int i = 0; i < this->nrows; ++i)
    {
        const int begin = this->hCsrMatOffs[static_cast<size_t>(i)];
        const int end = this->hCsrMatOffs[static_cast<size_t>(i) + 1];
        for (int k = begin; k < end; ++k)
        {
            const int col = this->hCsrMatInds[static_cast<size_t>(k)];
            const double val = this->hCsrMatVals[static_cast<size_t>(k)];
            if (col >= 0 && col < this->ncolsOriginal && val > this->env->getPxTolerance())
            {
                ctx.rowsByColumn[static_cast<size_t>(col)].push_back(i);
            }
        }
    }

    std::vector<std::unique_ptr<IColumnPreprocessRule>> rules = makeColumnPreprocessRules(this->env->getPreprocessColumnStrategies());
    int removedByRules = 0;
    for (const std::unique_ptr<IColumnPreprocessRule> &rule : rules)
    {
        removedByRules += rule->apply(ctx, this->env->getPxTolerance());
    }

    if (removedByRules <= 0)
    {
        return CODE_SUCCESSFUL;
    }

    std::vector<int> newActiveToInput;
    std::vector<int> newToOld;
    newActiveToInput.reserve(static_cast<size_t>(ctx.ncols));
    newToOld.reserve(static_cast<size_t>(ctx.ncols));
    std::vector<int> oldToNew(static_cast<size_t>(ctx.ncols), -1);
    for (int oldCol = 0; oldCol < ctx.ncols; ++oldCol)
    {
        if (!ctx.active[static_cast<size_t>(oldCol)])
        {
            continue;
        }
        const int newCol = static_cast<int>(newActiveToInput.size());
        oldToNew[static_cast<size_t>(oldCol)] = newCol;
        newActiveToInput.push_back(this->hActiveToInputCols[static_cast<size_t>(oldCol)]);
        newToOld.push_back(oldCol);
    }

    if (newActiveToInput.empty())
    {
        this->env->getLogger()->log(LOG_INFO, "Dominance rules removed all columns; keeping original model");
        return CODE_SUCCESSFUL;
    }

    rebuildCsrAfterRemoval(oldToNew, newToOld, newActiveToInput, ctx.costs.data());

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaNodeSparse::applyCostDrivenReduction()
{
    if (this->nrows <= 0 || this->ncolsOriginal <= 0 || this->hObjDns.empty() ||
        this->hCsrMatInds.empty() || this->hCsrMatOffs.empty() || this->hCsrMatVals.empty())
    {
        return CODE_SUCCESSFUL;
    }

    initActiveColTracking();

    ColumnPreprocessContext ctx;
    ctx.nrows = this->nrows;
    ctx.ncols = this->ncolsOriginal;
    ctx.rowsByColumn.assign(static_cast<size_t>(ctx.ncols), std::vector<int>());
    ctx.costs.assign(this->hObjDns.begin(), this->hObjDns.begin() + this->ncolsOriginal);
    ctx.active.assign(static_cast<size_t>(ctx.ncols), 1);
    if (this->env->getPreprocessTimeLimitSeconds() > 0.0)
    {
        ctx.deadline = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(this->env->getPreprocessTimeLimitSeconds()));
    }

    for (int i = 0; i < this->nrows; ++i)
    {
        const int begin = this->hCsrMatOffs[static_cast<size_t>(i)];
        const int end = this->hCsrMatOffs[static_cast<size_t>(i) + 1];
        for (int k = begin; k < end; ++k)
        {
            const int col = this->hCsrMatInds[static_cast<size_t>(k)];
            const double val = this->hCsrMatVals[static_cast<size_t>(k)];
            if (col >= 0 && col < this->ncolsOriginal && val > this->env->getPxTolerance())
            {
                ctx.rowsByColumn[static_cast<size_t>(col)].push_back(i);
            }
        }
    }

    std::vector<std::unique_ptr<IColumnPreprocessRule>> rules = makeColumnPreprocessRules("cost_driven_replacement");
    int removedByRules = 0;
    for (const std::unique_ptr<IColumnPreprocessRule> &rule : rules)
    {
        removedByRules += rule->apply(ctx, this->env->getPxTolerance());
    }

    if (removedByRules <= 0)
    {
        return CODE_SUCCESSFUL;
    }

    std::vector<int> newActiveToInput;
    std::vector<int> newToOld;
    newActiveToInput.reserve(static_cast<size_t>(ctx.ncols));
    newToOld.reserve(static_cast<size_t>(ctx.ncols));
    std::vector<int> oldToNew(static_cast<size_t>(ctx.ncols), -1);
    for (int oldCol = 0; oldCol < ctx.ncols; ++oldCol)
    {
        if (!ctx.active[static_cast<size_t>(oldCol)])
        {
            continue;
        }
        const int newCol = static_cast<int>(newActiveToInput.size());
        oldToNew[static_cast<size_t>(oldCol)] = newCol;
        newActiveToInput.push_back(this->hActiveToInputCols[static_cast<size_t>(oldCol)]);
        newToOld.push_back(oldCol);
    }

    if (newActiveToInput.empty())
    {
        this->env->getLogger()->log(LOG_INFO, "Cost-driven replacement removed all columns; keeping original model");
        return CODE_SUCCESSFUL;
    }

    rebuildCsrAfterRemoval(oldToNew, newToOld, newActiveToInput, ctx.costs.data());

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaNodeSparse::releaseModelOnDevice()
{
    if (this->dCsrMatInds)
    {
        checkCudaErrors(cudaFree(this->dCsrMatInds));
        this->dCsrMatInds = nullptr;
    }
    if (this->dCsrMatOffs)
    {
        checkCudaErrors(cudaFree(this->dCsrMatOffs));
        this->dCsrMatOffs = nullptr;
    }
    if (this->dCsrMatVals)
    {
        checkCudaErrors(cudaFree(this->dCsrMatVals));
        this->dCsrMatVals = nullptr;
    }

    if (this->dCsrMatTransInds)
    {
        checkCudaErrors(cudaFree(this->dCsrMatTransInds));
        this->dCsrMatTransInds = nullptr;
    }
    if (this->dCsrMatTransOffs)
    {
        checkCudaErrors(cudaFree(this->dCsrMatTransOffs));
        this->dCsrMatTransOffs = nullptr;
    }
    if (this->dCsrMatTransVals)
    {
        checkCudaErrors(cudaFree(this->dCsrMatTransVals));
        this->dCsrMatTransVals = nullptr;
    }

    if (this->dObjDns)
    {
        checkCudaErrors(cudaFree(this->dObjDns));
        this->dObjDns = nullptr;
    }
    if (this->dRhsDns)
    {
        checkCudaErrors(cudaFree(this->dRhsDns));
        this->dRhsDns = nullptr;
    }

    if (this->matDescr)
    {
        checkCudaErrors(cusparseDestroySpMat(this->matDescr));
        this->matDescr = nullptr;
    }
    if (this->matTransDescr)
    {
        checkCudaErrors(cusparseDestroySpMat(this->matTransDescr));
        this->matTransDescr = nullptr;
    }
    if (this->objDescr)
    {
        checkCudaErrors(cusparseDestroyDnVec(this->objDescr));
        this->objDescr = nullptr;
    }
    if (this->rhsDescr)
    {
        checkCudaErrors(cusparseDestroyDnVec(this->rhsDescr));
        this->rhsDescr = nullptr;
    }

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaNodeSparse::setInitValues()
{
    return CODE_SUCCESSFUL;
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

    return CODE_SUCCESSFUL;
}