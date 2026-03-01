
#include "model_reader.h"
#include "sypha_node_sparse.h"

/**
 *  Read a Set Covering model and convert it to standard form.
 *  A | S = b
 */
SyphaStatus model_reader_read_scp_file_sparse_coo(SyphaNodeSparse &node, std::string inputFilePath)
{
    int i, j, idx, currColNumber;
    double val;
    SyphaLogger *log = node.env->getLogger();

    log->log(LOG_DEBUG, "Scanning SCP model (sparse COO) at %s", inputFilePath.c_str());
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");
    if (inputFileHandler == nullptr)
    {
        log->log(LOG_ERROR, "Failed to open input file: %s", inputFilePath.c_str());
        return CODE_GENERIC_ERROR;
    }

    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        log->log(LOG_ERROR, "Failed to parse SCP model dimensions (fscanf)");
        return CODE_GENERIC_ERROR;
    }

    node.hObjDns.assign(static_cast<size_t>(node.ncols + node.nrows), 0.0);
    node.hRhsDns.assign(static_cast<size_t>(node.nrows), 0.0);

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

            node.hCooMat.push_back(SyphaCOOEntry(i, idx - 1, 1.0));
        }
        node.hCooMat.push_back(SyphaCOOEntry(i, node.ncols + i, -1.0));
    }

    fclose(inputFileHandler);

    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
    }

    node.nnz = node.hCooMat.size();
    node.ncolsOriginal = node.ncols;
    node.ncolsInputOriginal = node.ncolsOriginal;
    node.hActiveToInputCols.clear();
    node.hActiveToInputCols.reserve(static_cast<size_t>(node.ncolsOriginal));
    for (int col = 0; col < node.ncolsOriginal; ++col)
    {
        node.hActiveToInputCols.push_back(col);
    }
    node.ncols = node.ncols + node.nrows;

    log->log(LOG_DEBUG, "Model: SCP (sparse COO), %d non-zeros", (node.nnz - node.nrows));

    return CODE_SUCCESSFUL;
}

/**
 *  Read a Set Covering model and convert it to standard form.
 *  A | S = b
 */
SyphaStatus model_reader_read_scp_file_sparse_csr(SyphaNodeSparse &node, std::string inputFilePath)
{
    int i, j, idx, currColNumber;
    double val;
    SyphaLogger *log = node.env->getLogger();

    log->log(LOG_DEBUG, "Scanning SCP model (sparse CSR) at %s", inputFilePath.c_str());
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");
    if (inputFileHandler == nullptr)
    {
        log->log(LOG_ERROR, "Failed to open input file: %s", inputFilePath.c_str());
        return CODE_GENERIC_ERROR;
    }

    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        log->log(LOG_ERROR, "Failed to parse SCP model dimensions (fscanf)");
        return CODE_GENERIC_ERROR;
    }

    node.hObjDns.assign(static_cast<size_t>(node.ncols + node.nrows), 0.0);
    node.hRhsDns.assign(static_cast<size_t>(node.nrows), 0.0);

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
    node.hCsrMatOffs.push_back(0);
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
            node.hCsrMatInds.push_back(idx - 1);
            node.hCsrMatVals.push_back(1.0);
        }

        node.hCsrMatInds.push_back(node.ncols + i);
        node.hCsrMatVals.push_back(-1.0);

        node.hCsrMatOffs.push_back(node.hCsrMatVals.size());
    }

    fclose(inputFileHandler);

    log->log(LOG_TRACE, "Adding right-hand sides");
    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
    }

    node.nnz = node.hCsrMatVals.size();
    node.ncolsOriginal = node.ncols;
    node.ncolsInputOriginal = node.ncolsOriginal;
    node.hActiveToInputCols.clear();
    node.hActiveToInputCols.reserve(static_cast<size_t>(node.ncolsOriginal));
    for (int col = 0; col < node.ncolsOriginal; ++col)
    {
        node.hActiveToInputCols.push_back(col);
    }
    node.ncols = node.ncols + node.nrows;

    log->log(LOG_DEBUG, "Model: SCP (sparse CSR), %d non-zeros", (node.nnz - node.nrows));

    return CODE_SUCCESSFUL;
}
