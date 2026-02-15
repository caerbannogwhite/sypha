
#include "model_reader.h"

SyphaStatus model_reader_read_scp_file_dense(SyphaNodeDense &node, string inputFilePath)
{
    int i = 0, j = 0, idx = 0, currColNumber = 0, ncolsAS = 0, nnz = 0;
    double val = 0.0;
    char message[1024];

    node.env->logger("Start scanning SCP model at " + inputFilePath, "INFO", 10);
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of sets and elements
    // SCP file format: num_sets num_elements
    // In standard form: sets are columns (variables), elements are rows (constraints)
    if (!fscanf(inputFileHandler, "%d %d", &node.ncols, &node.nrows))
    {
        node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
        return CODE_GENERIC_ERROR;
    }

    sprintf(message, "Original model has %d rows and %d columns", node.nrows, node.ncols);
    node.env->logger(message, "INFO", 15);

    ncolsAS = node.ncols + node.nrows;
    node.hObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.hRhsDns = (double *)calloc(node.nrows, sizeof(double));
    node.hMatDns = (double *)calloc(node.nrows * ncolsAS, sizeof(double));

    // read objective
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }
        node.hObjDns[j] = val;
    }

    // read rows
    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }

        nnz += currColNumber;
        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
                return CODE_GENERIC_ERROR;
            }
            node.hMatDns[i * ncolsAS + idx - 1] = 1.0;
        }
    }

    fclose(inputFileHandler);

    // add elements in S and rhs
    nnz += node.nrows;
    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
        node.hMatDns[i * ncolsAS + node.ncols + i] = -1.0;
    }

    // update num cols
    node.nnz = nnz;
    node.ncolsOriginal = node.ncols; // Save original column count before adding slacks
    node.ncols = ncolsAS;
    sprintf(message, "Model: SCP (dense), %d non-zeros", (nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

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
    char message[1024];

    node.env->logger("Start scanning SCP model at " + inputFilePath, "INFO", 10);
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of sets and elements
    // SCP file format: num_sets num_elements
    // In standard form: sets are columns (variables), elements are rows (constraints)
    if (!fscanf(inputFileHandler, "%d %d", &node.ncols, &node.nrows))
    {
        node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
        return CODE_GENERIC_ERROR;
    }

    node.hObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.hRhsDns = (double *)calloc(node.nrows, sizeof(double));

    // read objective
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }
        node.hObjDns[j] = val;
    }

    // read rows
    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
                return CODE_GENERIC_ERROR;
            }

            node.hCooMat->push_back(SyphaCOOEntry(i, idx - 1, 1.0));
        }
        node.hCooMat->push_back(SyphaCOOEntry(i, node.ncols + i, -1.0));
    }

    fclose(inputFileHandler);

    // add S objective and right hand sides
    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
    }

    node.nnz = node.hCooMat->size();
    node.ncolsOriginal = node.ncols; // Save original column count before adding slacks
    node.ncols = node.ncols + node.nrows;

    sprintf(message, "Model: SCP (sparse COO), %d non-zeros", (node.nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

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
    char message[1024];

    node.env->logger("Start scanning SCP model at " + inputFilePath, "INFO", 10);
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of sets and elements
    // SCP file format: num_sets num_elements
    // In standard form: sets are columns (variables), elements are rows (constraints)
    if (!fscanf(inputFileHandler, "%d %d", &node.ncols, &node.nrows))
    {
        node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
        return CODE_GENERIC_ERROR;
    }

    node.hObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.hRhsDns = (double *)calloc(node.nrows, sizeof(double));

    // read objective
    sprintf(message, "Original model has %d rows and %d columns", node.nrows, node.ncols);
    node.env->logger(message, "INFO", 15);
    node.env->logger("Start scanning model objective", "INFO", 20);
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }
        node.hObjDns[j] = val;
    }

    // Skip to end of current line after reading costs
    // (costs and first set definition may share a line)
    int c;
    while ((c = fgetc(inputFileHandler)) != '\n' && c != EOF)
    {
        // discard remaining characters on the line
    }

    // read sets and build matrix
    // File has one line per set, describing which elements it covers
    // We need to build matrix A where A[elem][set] = 1 if set covers elem
    node.env->logger("Start scanning sets", "INFO", 20);

    // Use vector to accumulate rows, since we're reading by columns (sets)
    vector<vector<pair<int, double>>> rows_data(node.nrows);

    for (j = 0; j < node.ncols; ++j) // iterate over sets
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }

        for (int k = 0; k < currColNumber; ++k)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
                return CODE_GENERIC_ERROR;
            }
            // idx is 1-indexed element, this set covers it
            // So matrix[idx-1][j] = 1.0
            rows_data[idx - 1].push_back(make_pair(j, 1.0));
        }
    }

    // Now build CSR format from rows_data, and add slack variables
    node.hCsrMatOffs->push_back(0);
    for (i = 0; i < node.nrows; ++i)
    {
        // Add entries for this row (element constraint)
        for (const auto &entry : rows_data[i])
        {
            node.hCsrMatInds->push_back(entry.first);
            node.hCsrMatVals->push_back(entry.second);
        }
        // Add slack variable entry
        node.hCsrMatInds->push_back(node.ncols + i);
        node.hCsrMatVals->push_back(-1.0);

        node.hCsrMatOffs->push_back(node.hCsrMatVals->size());
    }

    fclose(inputFileHandler);

    node.env->logger("Adding right hand sides", "INFO", 30);
    // set right hand sides
    for (i = 0; i < node.nrows; ++i)
    {
        node.hRhsDns[i] = 1.0;
    }

    node.nnz = node.hCsrMatVals->size();
    node.ncolsOriginal = node.ncols; // Save original column count before adding slacks
    node.ncols = node.ncols + node.nrows;

    sprintf(message, "Model: SCP (sparse CSR), %d non-zeros", (node.nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

    return CODE_SUCCESFULL;
}
