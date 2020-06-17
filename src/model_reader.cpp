
#include "model_reader.h"

SyphaStatus model_reader_read_scp_file_dense(SyphaNodeDense &node, string inputFilePath)
{
    int i = 0, j = 0, idx = 0, currColNumber = 0, ncolsAS = 0, nnz = 0;
    double val = 0.0;
    char message[1024];

    node.env->logger("Start scanning SCP model at " + inputFilePath, "INFO", 10);
    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of rows and columns
    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
        return CODE_ERROR;
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
            return CODE_ERROR;
        }
        node.h_ObjDns[j] = val;
    }

    // read rows
    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
            return CODE_ERROR;
        }

        nnz += currColNumber;
        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                node.env->logger("model_reader_read_scp_file_dense: fscanf failed.", "ERROR", 0);
                return CODE_ERROR;
            }
            node.h_MatDns[i*ncolsAS + idx - 1] = 1.0;
        }
    }

    fclose(inputFileHandler);

    // add elements in S and rhs
    nnz += node.nrows;
    for (i = 0; i < node.nrows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
        node.h_MatDns[i*ncolsAS + node.ncols + i] = -1.0;
    }
    
    // update num cols
    node.nnz = nnz;
    node.ncols = ncolsAS;
    sprintf(message, "Successfully read SCP model (dense) with %d non zeros", (nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

    // printf("Mat (nnz: %d)\n", node.nnz);
    // for (int i = 0; i < node.nrows; ++i)
    // {
    //     for (int j = 0; j < node.ncols; ++j)
    //     {
    //         printf("%3.0lf ", node.h_MatDns[node.ncols * i + j]);
    //     }
    //     printf("\n");
    // }

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

    // read number of rows and columns
    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
        return CODE_ERROR;
    }

    node.h_ObjDns = (double *)calloc(node.ncols + node.nrows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.nrows, sizeof(double));

    // read objective
    for (j = 0; j < node.ncols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
            return CODE_ERROR;
        }
        node.h_ObjDns[j] = val;
    }

    // read rows
    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
            return CODE_ERROR;
        }

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                node.env->logger("model_reader_read_scp_file_sparse_coo: fscanf failed.", "ERROR", 0);
                return CODE_ERROR;
            }

            node.h_cooMat->push_back(SyphaCOOEntry(i, idx-1, 1.0));
        }
        node.h_cooMat->push_back(SyphaCOOEntry(i, node.ncols+i, -1.0));
    }

    fclose(inputFileHandler);

    // add S objective and right hand sides
    for (i = 0; i < node.nrows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
    }

    node.nnz = node.h_cooMat->size();
    node.ncols = node.ncols + node.nrows;

    sprintf(message, "Successfully read SCP model (sparse COO) with %d non zeros", (node.nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

    // std::cout << "mat\n";
    // for (auto it = node.h_cooMat->cbegin(); it != node.h_cooMat->cend(); ++it)
    //     std::cout << (*it).row << " " << (*it).col << " " << (*it).val << std::endl;
    // std::cout << std::endl;

    // std::cout << "obj\n";
    // for (i = 0; i < node.ncols; ++i)
    //     std::cout << node.h_ObjDns[i] << " ";
    // std::cout << std::endl << std::endl;

    // std::cout << "rhs\n";
    // for (i = 0; i < node.nrows; ++i)
    //     std::cout << node.h_RhsDns[i] << " ";
    // std::cout << std::endl << std::endl;

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

    // read number of rows and columns
    if (!fscanf(inputFileHandler, "%d %d", &node.nrows, &node.ncols))
    {
        node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
        return CODE_ERROR;
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
            return CODE_ERROR;
        }
        node.h_ObjDns[j] = val;
    }

    // read rows
    node.env->logger("Start scanning rows", "INFO", 20);
    node.h_csrMatOffs->push_back(0);
    for (i = 0; i < node.nrows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
            return CODE_ERROR;
        }

        // add 1 for the element in matrix S
        node.h_csrMatOffs->push_back(node.h_csrMatOffs->back() + currColNumber + 1);

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                node.env->logger("model_reader_read_scp_file_sparse_csr: fscanf failed.", "ERROR", 0);
                return CODE_ERROR;
            }
            node.h_csrMatInds->push_back(idx - 1);
            node.h_csrMatVals->push_back(1.0);
        }
        node.h_csrMatInds->push_back(node.ncols + i);
        node.h_csrMatVals->push_back(-1.0);
    }

    fclose(inputFileHandler);

    node.env->logger("Adding right hand sides", "INFO", 30);
    // set right hand sides
    for (i = 0; i < node.nrows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
    }

    node.nnz = node.h_csrMatVals->size();
    node.ncols = node.ncols + node.nrows;

    sprintf(message, "Successfully read SCP model (sparse CSR) with %d non zeros", (node.nnz - node.nrows));
    node.env->logger(message, "INFO", 10);

    // std::cout << "indeces\n";
    // for (auto it = node.h_csrMatInds->cbegin(); it != node.h_csrMatInds->cend(); ++it)
    //     std::cout << *it << " ";
    // std::cout << std::endl;

    // std::cout << "indeces ptr\n";
    // for (auto it = node.h_csrMatOffs->cbegin(); it != node.h_csrMatOffs->cend(); ++it)
    //     std::cout << *it << " ";
    // std::cout << std::endl;
    
    // std::cout << "vals\n";
    // for (auto it = node.h_csrMatVals->cbegin(); it != node.h_csrMatVals->cend(); ++it)
    //     std::cout << *it << " ";
    // std::cout << std::endl;

    // std::cout << "obj\n";
    // for (auto it = node.h_ObjDns->cbegin(); it != node.h_ObjDns->cend(); ++it)
    //     std::cout << *it << " ";
    // std::cout << std::endl;

    // std::cout << "rhs\n";
    // for (auto it = node.h_RhsDns->cbegin(); it != node.h_RhsDns->cend(); ++it)
    //     std::cout << *it << " ";
    // std::cout << std::endl;

    return CODE_SUCCESFULL;
}
