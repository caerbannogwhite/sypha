
#include "model_reader.h"

SyphaStatus model_reader_read_scp_file_dense(SyphaNodeDense &node, string inputFilePath)
{
    int i, j, idx, currColNumber, numColsAS;
    double val;

    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of rows and columns
    if (!fscanf(inputFileHandler, "%d %d", &node.numRows, &node.numCols))
    {
        perror("ERROR: readInstance on fscanf.");
        return CODE_ERROR;
    }

    numColsAS = node.numCols + node.numRows;
    node.h_ObjDns = (double *)calloc(node.numCols + node.numRows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.numRows, sizeof(double));
    node.h_MatDns = (double *)calloc(node.numRows * numColsAS, sizeof(double));

    // read objective
    for (j = 0; j < node.numCols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }
        node.h_ObjDns[j] = val;
    }

    // read rows
    for (i = 0; i < node.numRows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            node.h_MatDns[i*numColsAS + idx - 1] = 1.0;
        }
    }

    fclose(inputFileHandler);

    // add elements in S and rhs
    for (i = 0; i < node.numRows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
        node.h_MatDns[i*numColsAS + node.numCols + i] = -1.0;
    }

    // update num cols
    node.numCols = numColsAS;

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

    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of rows and columns
    if (!fscanf(inputFileHandler, "%d %d", &node.numRows, &node.numCols))
    {
        perror("ERROR: readInstance on fscanf.");
        return CODE_ERROR;
    }

    node.h_ObjDns = (double *)calloc(node.numCols + node.numRows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.numRows, sizeof(double));

    // read objective
    for (j = 0; j < node.numCols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }
        node.h_ObjDns[j] = val;
    }

    // read rows
    for (i = 0; i < node.numRows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }

            node.h_cooMat->push_back(SyphaCOOEntry(i, idx-1, 1.0));
        }
        node.h_cooMat->push_back(SyphaCOOEntry(i, node.numCols+i, -1.0));
    }

    fclose(inputFileHandler);

    // add S objective and right hand sides
    for (i = 0; i < node.numRows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
    }

    node.numCols = node.numCols + node.numRows;

    /*std::cout << "mat\n";
    for (auto it = node.h_cooMat->cbegin(); it != node.h_cooMat->cend(); ++it)
        std::cout << (*it).row << " " << (*it).col << " " << (*it).val << std::endl;
    std::cout << std::endl;

    std::cout << "obj\n";
    for (i = 0; i < node.numCols; ++i)
        std::cout << node.h_ObjDns[i] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "rhs\n";
    for (i = 0; i < node.numRows; ++i)
        std::cout << node.h_RhsDns[i] << " ";
    std::cout << std::endl << std::endl;*/

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

    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    // read number of rows and columns
    if (!fscanf(inputFileHandler, "%d %d", &node.numRows, &node.numCols))
    {
        perror("ERROR: readInstance on fscanf.");
        return CODE_ERROR;
    }

    node.h_ObjDns = (double *)calloc(node.numCols + node.numRows, sizeof(double));
    node.h_RhsDns = (double *)calloc(node.numRows, sizeof(double));

    // read objective
    for (j = 0; j < node.numCols; ++j)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }
        node.h_ObjDns[j] = val;
    }

    // read rows
    node.h_csrMatIndPtrs->push_back(0);
    for (i = 0; i < node.numRows; ++i)
    {
        if (!fscanf(inputFileHandler, "%d", &currColNumber))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }

        // add 1 for the element in matrix S
        node.h_csrMatIndPtrs->push_back(node.h_csrMatIndPtrs->back() + currColNumber + 1);

        for (j = 0; j < currColNumber; ++j)
        {
            if (!fscanf(inputFileHandler, "%d", &idx))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            node.h_csrMatIndices->push_back(idx - 1);
            node.h_csrMatVals->push_back(1.0);
        }
        node.h_csrMatIndices->push_back(node.numCols + i);
        node.h_csrMatVals->push_back(-1.0);
    }

    fclose(inputFileHandler);

    // add S objective and right hand sides
    for (i = 0; i < node.numRows; ++i)
    {
        node.h_RhsDns[i] = 1.0;
    }

    node.numCols = node.numCols + node.numRows;

    /*std::cout << "indeces\n";
    for (auto it = node.h_csrMatIndices->cbegin(); it != node.h_csrMatIndices->cend(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;

    std::cout << "indeces ptr\n";
    for (auto it = node.h_csrMatIndPtrs->cbegin(); it != node.h_csrMatIndPtrs->cend(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;
    
    std::cout << "vals\n";
    for (auto it = node.h_csrMatVals->cbegin(); it != node.h_csrMatVals->cend(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;

    std::cout << "obj\n";
    for (auto it = node.h_ObjDns->cbegin(); it != node.h_ObjDns->cend(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;

    std::cout << "rhs\n";
    for (auto it = node.h_RhsDns->cbegin(); it != node.h_RhsDns->cend(); ++it)
        std::cout << *it << " ";
    std::cout << std::endl;*/

    return CODE_SUCCESFULL;
}
