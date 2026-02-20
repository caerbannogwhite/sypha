
#include "model_reader.h"

SyphaStatus model_reader_read_scp_file_dense(SyphaNodeDense &node, string inputFilePath)
{
    int i = 0, j = 0, idx = 0, currColNumber = 0, ncolsAS = 0, nnz = 0;
    double val = 0.0;
    SyphaLogger *log = node.env->getLogger();

    log->log(LOG_DEBUG, "Scanning SCP model (dense) at %s", inputFilePath.c_str());
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");
    if (inputFileHandler == NULL)
    {
        log->log(LOG_ERROR, "Failed to open input file: %s", inputFilePath.c_str());
        return CODE_GENERIC_ERROR;
    }

    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        log->log(LOG_ERROR, "Failed to parse SCP model dimensions (fscanf)");
        return CODE_GENERIC_ERROR;
    }

    log->log(LOG_TRACE, "Original model: %d rows, %d columns", node.nrows, node.ncols);

    ncolsAS = node.ncols + node.nrows;
    node.hObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.hRhsDns = (double *)calloc(node.nrows, sizeof(double));
    node.hMatDns = (double *)calloc(node.nrows * ncolsAS, sizeof(double));

    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            log->log(LOG_ERROR, "Failed to parse objective coefficient at column %d", j);
            return CODE_GENERIC_ERROR;
        }
        node.hObjDns[j] = val;
    }

    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            log->log(LOG_ERROR, "Failed to parse row %d column count", i);
            return CODE_GENERIC_ERROR;
        }

        nnz += currColNumber;
        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                log->log(LOG_ERROR, "Failed to parse column index at row %d", i);
                return CODE_GENERIC_ERROR;
            }
            node.hMatDns[i * ncolsAS + idx - 1] = 1.0;
        }
    }

    fclose(inputFileHandler);

    nnz += node.nrows;
    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
        node.hMatDns[i * ncolsAS + node.ncols + i] = -1.0;
    }

    node.nnz = nnz;
    node.ncolsOriginal = node.ncols;
    node.ncols = ncolsAS;
    log->log(LOG_DEBUG, "Model: SCP (dense), %d non-zeros", (nnz - node.nrows));

    return CODE_SUCCESFULL;
}

/**
 *  Read a Set Covering model and convert it to standard form.
 *  A | S = b
 */
SyphaStatus model_reader_read_scp_file_sparse_coo(SyphaNodeSparse &node, string inputFilePath)
{
    int i, j, idx, currColNumber;
    double val;
    SyphaLogger *log = node.env->getLogger();

    log->log(LOG_DEBUG, "Scanning SCP model (sparse COO) at %s", inputFilePath.c_str());
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");
    if (inputFileHandler == NULL)
    {
        log->log(LOG_ERROR, "Failed to open input file: %s", inputFilePath.c_str());
        return CODE_GENERIC_ERROR;
    }

    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        log->log(LOG_ERROR, "Failed to parse SCP model dimensions (fscanf)");
        return CODE_GENERIC_ERROR;
    }

    node.hObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.hRhsDns = (double *)calloc(node.nrows, sizeof(double));

    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            log->log(LOG_ERROR, "Failed to parse objective coefficient at column %d", j);
            return CODE_GENERIC_ERROR;
        }
        node.hObjDns[j] = val;
    }

    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            log->log(LOG_ERROR, "Failed to parse row %d column count", i);
            return CODE_GENERIC_ERROR;
        }

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                log->log(LOG_ERROR, "Failed to parse column index at row %d", i);
                return CODE_GENERIC_ERROR;
            }

            node.hCooMat->push_back(SyphaCOOEntry(i, idx - 1, 1.0));
        }
        node.hCooMat->push_back(SyphaCOOEntry(i, node.ncols + i, -1.0));
    }

    fclose(inputFileHandler);

    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
    }

    node.nnz = node.hCooMat->size();
    node.ncolsOriginal = node.ncols;
    node.ncolsInputOriginal = node.ncolsOriginal;
    node.hActiveToInputCols->clear();
    node.hActiveToInputCols->reserve((size_t)node.ncolsOriginal);
    for (int col = 0; col < node.ncolsOriginal; ++col)
    {
        node.hActiveToInputCols->push_back(col);
    }
    node.ncols = node.ncols + node.nrows;

    log->log(LOG_DEBUG, "Model: SCP (sparse COO), %d non-zeros", (node.nnz - node.nrows));

    return CODE_SUCCESFULL;
}

/**
 *  Read a Set Covering model and convert it to standard form.
 *  A | S = b
 */
SyphaStatus model_reader_read_scp_file_sparse_csr(SyphaNodeSparse &node, string inputFilePath)
{
    int i, j, idx, currColNumber;
    double val;
    SyphaLogger *log = node.env->getLogger();

    log->log(LOG_DEBUG, "Scanning SCP model (sparse CSR) at %s", inputFilePath.c_str());
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");
    if (inputFileHandler == NULL)
    {
        log->log(LOG_ERROR, "Failed to open input file: %s", inputFilePath.c_str());
        return CODE_GENERIC_ERROR;
    }

    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        log->log(LOG_ERROR, "Failed to parse SCP model dimensions (fscanf)");
        return CODE_GENERIC_ERROR;
    }

    node.hObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.hRhsDns = (double *)calloc(node.nrows, sizeof(double));

    log->log(LOG_TRACE, "Original model: %d rows, %d columns", node.nrows, node.ncols);
    log->log(LOG_TRACE, "Scanning model objective");
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            log->log(LOG_ERROR, "Failed to parse objective coefficient at column %d", j);
            return CODE_GENERIC_ERROR;
        }
        node.hObjDns[j] = val;
    }

    log->log(LOG_TRACE, "Scanning rows");
    node.hCsrMatOffs->push_back(0);
    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            log->log(LOG_ERROR, "Failed to parse row %d column count", i);
            return CODE_GENERIC_ERROR;
        }

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                log->log(LOG_ERROR, "Failed to parse column index at row %d", i);
                return CODE_GENERIC_ERROR;
            }
            node.hCsrMatInds->push_back(idx - 1);
            node.hCsrMatVals->push_back(1.0);
        }

        node.hCsrMatInds->push_back(node.ncols + i);
        node.hCsrMatVals->push_back(-1.0);

        node.hCsrMatOffs->push_back(node.hCsrMatVals->size());
    }

    fclose(inputFileHandler);

    log->log(LOG_TRACE, "Adding right-hand sides");
    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
    }

    node.nnz = node.hCsrMatVals->size();
    node.ncolsOriginal = node.ncols;
    node.ncolsInputOriginal = node.ncolsOriginal;
    node.hActiveToInputCols->clear();
    node.hActiveToInputCols->reserve((size_t)node.ncolsOriginal);
    for (int col = 0; col < node.ncolsOriginal; ++col)
    {
        node.hActiveToInputCols->push_back(col);
    }
    node.ncols = node.ncols + node.nrows;

    log->log(LOG_DEBUG, "Model: SCP (sparse CSR), %d non-zeros", (node.nnz - node.nrows));

    return CODE_SUCCESFULL;
}
