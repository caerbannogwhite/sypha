
#include "model_reader.h"
#include "sypha_node.h"

SyphaStatus model_reader_read_scp_file_dense(SyphaNode &node, string inputFilePath)
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


SyphaStatus model_reader_scp_model_to_standard_dense(SyphaNode &node, string inputFilePath)
{

    return CODE_SUCCESFULL;
}

SyphaStatus model_reader_read_scp_file_sparse(SyphaNode &node, string inputFilePath)
{
    int lineCounter = 0, currColNumber, colIdx = 0, rowIdx = 0, num;
    bool newRowFlag = false;

    FILE *inputFileHandler = fopen(inputFilePath.c_str(), "r");

    /*while (!feof(inputFileHandler))
    {
        // first row: number of rows and cols
        if (lineCounter == 0)
        {
            if (!fscanf(inputFileHandler, "%d %d", &inst.nrows, &inst.ncols))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }

            inst.hostObj = (double *)calloc(inst.ncols, sizeof(double));
            inst.hostMat = (double *)calloc(inst.nrows * inst.ncols, sizeof(double));
        }

        // costs
        else if (lineCounter > 0 && lineCounter <= inst.ncols)
        {
            if (!fscanf(inputFileHandler, "%d", &num))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            inst.hostObj[colIdx++] = num;

            if (lineCounter == inst.ncols)
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
            inst.hostMat[rowIdx * inst.ncols + num - 1] = 1;
            ++colIdx;

            if (currColNumber == colIdx)
            {
                colIdx = 0;
                ++rowIdx;
                newRowFlag = true;
            }
        }

        ++lineCounter;
    }*/

    fclose(inputFileHandler);
    return CODE_SUCCESFULL;
}

SyphaStatus model_reader_scp_model_to_standard_sparse(SyphaNode &node, string inputFilePath)
{

    return CODE_SUCCESFULL;
}