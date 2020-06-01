
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
    int lineCounter = 0, currColNumber, num;
    int nonZeroCounter = 0;
    bool newRowFlag = false;

    vector<int> indeces = new vector();
    vector<int> indPtrs = new vector();

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

            node.h_ObjDns = (double *)calloc(node.numCols + node.numRows, sizeof(double));
        }

        // costs: lineCounter counts the number of objective entries (num cols)
        else if (lineCounter > 0 && lineCounter <= node.numCols)
        {
            if (!fscanf(inputFileHandler, "%d", &num))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            node.h_ObjDns[colIdx++] = num;

            if (lineCounter == node.numCols)
            {
                newRowFlag = true;
                indPtrs.push_back(0);
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
            indPtrs.push_back(currColNumber);
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
            indeces.push_back(num - 1);
            ++nonZeroCounter;

            if (currColNumber == colIdx)
            {
                newRowFlag = true;
                indPtrs.push_back(nonZeroCounter);
            }
        }

        ++lineCounter;
    }

    fclose(inputFileHandler);
    return CODE_SUCCESFULL;
}
