
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
    node.h_ObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.nrows, sizeof(double));
    node.h_MatDns = (double *)calloc(node.nrows * ncolsAS, sizeof(double));

    // read objective
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }
        node.h_ObjDns[j] = val;
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
            node.h_MatDns[i * ncolsAS + idx - 1] = 1.0;
        }
    }

    fclose(inputFileHandler);

    // add elements in S and rhs
    nnz += node.nrows;
    for (i = 0; i < node.nrows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
        node.h_MatDns[i * ncolsAS + node.ncols + i] = -1.0;
    }

    // update num cols
    node.nnz = nnz;
    node.ncolsOriginal = node.ncols; // Save original column count before adding slacks
    node.ncols = ncolsAS;
    sprintf(message, "Successfully read SCP model (dense) with %d non zeros", (nnz - node.nrows));
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

    node.h_ObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.nrows, sizeof(double));

    // read objective
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
            return CODE_GENERIC_ERROR;
        }
        node.h_ObjDns[j] = val;
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

            node.h_cooMat->push_back(SyphaCOOEntry(i, idx - 1, 1.0));
        }
        node.h_cooMat->push_back(SyphaCOOEntry(i, node.ncols + i, -1.0));
    }

    fclose(inputFileHandler);

    // add S objective and right hand sides
    for (i = 0; i < node.nrows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
    }

    node.nnz = node.h_cooMat->size();
    node.ncolsOriginal = node.ncols; // Save original column count before adding slacks
    node.ncols = node.ncols + node.nrows;

    sprintf(message, "Successfully read SCP model (sparse COO) with %d non zeros", (node.nnz - node.nrows));
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

    node.h_ObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.nrows, sizeof(double));

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
        node.h_ObjDns[j] = val;
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
    node.h_csrMatOffs->push_back(0);
    for (i = 0; i < node.nrows; ++i)
    {
        // Add entries for this row (element constraint)
        for (const auto &entry : rows_data[i])
        {
            node.h_csrMatInds->push_back(entry.first);
            node.h_csrMatVals->push_back(entry.second);
        }
        // Add slack variable entry
        node.h_csrMatInds->push_back(node.ncols + i);
        node.h_csrMatVals->push_back(-1.0);

        node.h_csrMatOffs->push_back(node.h_csrMatVals->size());
    }

    fclose(inputFileHandler);

    node.env->logger("Adding right hand sides", "INFO", 30);
    // set right hand sides
    for (i = 0; i < node.nrows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
    }

    node.nnz = node.h_csrMatVals->size();
    node.ncolsOriginal = node.ncols; // Save original column count before adding slacks
    node.ncols = node.ncols + node.nrows;

    sprintf(message, "Successfully read SCP model (sparse CSR) with %d non zeros", (node.nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

    return CODE_SUCCESFULL;
}
