
#include "model_reader.h"

SyphaStatus model_reader_read_scp_file_dense(SyphaNodeDense &node, string inputFilePath)
{
    int currColNumber, num;
    int lineCounter = 0;
    int colIdx = 0;
    int rowIdx = 0;
    bool newRowFlag = false;

    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    while (!feof(inputFileHandler))
    {
        // first row: number of rows and cols
        if (lineCounter == 0)
        {
            if (!fscanf(inputFileHandler, "%d %d", &node.numRows, &node.numCols))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }

            node.h_ObjDns = (double *)calloc(node.numCols, sizeof(double));
            node.h_MatDns = (double *)calloc(node.numRows * node.numCols, sizeof(double));
        }

        // costs
        else if ((lineCounter > 0) && (lineCounter <= node.numCols))
        {
            if (!fscanf(inputFileHandler, "%d", &num))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            node.h_ObjDns[colIdx++] = num;

            if (lineCounter == node.numCols)
            {
                colIdx = 0;
                rowIdx = 0;
                newRowFlag = true;
            }
        }

        // new row
        else if (newRowFlag)
        {
            if (!fscanf(inputFileHandler, "%d", &currColNumber))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            newRowFlag = false;
        }

        // entries
        else
        {
            // num: index of the column whose coefficient must be 1, 1 <= index <= num_cols
            if (!fscanf(inputFileHandler, "%d", &num))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            node.h_MatDns[rowIdx * node.numCols + num - 1] = 1;
            ++colIdx;

            if (currColNumber == colIdx)
            {
                colIdx = 0;
                ++rowIdx;
                newRowFlag = true;
            }
        }

        ++lineCounter;
    }

    fclose(inputFileHandler);
    return CODE_SUCCESFULL;
}

/**
 *  Read a Set Covering model and convert it to standard form.
 *  A | S = b 
 */
SyphaStatus model_reader_read_scp_file_sparse(SyphaNodeSparse &node, string inputFilePath)
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

    // read objective
    for (i = 0; i < node.numCols; ++i)
    {
        if (!fscanf(inputFileHandler, "%lf", &val))
        {
            perror("ERROR: readInstance on fscanf.");
            return CODE_ERROR;
        }
        node.h_ObjDns->push_back(val);
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
    for (int i = 0; i < node.numRows; ++i)
    {
        node.h_ObjDns->push_back(0.0);
        node.h_RhsDns->push_back(1.0);
    }

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
