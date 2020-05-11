
#include "sypha_node.h"

SyphaNode::SyphaNode(SyphaEnvironment &env)
{
    this->env = env;
    this->sparse = env.sparse;

    this->numCols = 0;
    this->numRows = 0;
    this->numNonZero = 0;

    this->objectiveValue = 0.0;
}

bool SyphaNode::isSparse()
{
    return this->sparse;
}

int SyphaNode::getNumCols()
{
    return this->numCols;
}

int SyphaNode::getNumRows()
{
    return this->numRows;
}

int SyphaNode::getNumNonZero()
{
    return this->numRows;
}

double SyphaNode::getObjectiveValue()
{
    return this->objectiveValue;
}

SyphaStatus SyphaNode::solve()
{

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNode::importModel()
{
    if (this->env.modelType == MODEL_TYPE_SCP)
    {
        if (this->sparse)
        {
            model_reader_read_scp_file_dense(*this, this->env.inputFilePath);
        } else {
            model_reader_read_scp_file_sparse(*this, this->env.inputFilePath);
        }
    } else {
        return CODE_MODEL_TYPE_NOT_FOUND;
    }
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNode::copyModelOnDevice()
{
    return CODE_SUCCESFULL;
}

SyphaStatus SyphaNode::convert2MySimplexForm()
{
    /*inst.localObj = (double *)realloc(inst.localObj, (inst.ncols + inst.nrows) * sizeof(double));
    inst.localMat = (double *)realloc(inst.localMat, (inst.ncols + inst.nrows) * inst.nrows * sizeof(double));

    int i, j;

    for (j = inst.ncols; j < inst.ncols + inst.nrows; ++j)
    {
        inst.localObj[j] = 0.0;
    }

    for (i = inst.nrows - 1; i >= 0; i--)
    {
        for (j = 0; j < inst.ncols; ++j)
        {
            inst.localMat[i * (inst.ncols + inst.nrows) + j] = -inst.localMat[i * (inst.ncols) + j];
        }
    }

    for (i = 0; i < inst.nrows; ++i)
    {
        for (j = inst.ncols; j < inst.ncols + inst.nrows; ++j)
        {
            inst.localMat[i * (inst.ncols + inst.nrows) + j] = i == (j - inst.ncols) ? 1.0 : 0.0;
        }
    }

    inst.ncols = inst.ncols + inst.nrows;*/
    return CODE_SUCCESFULL;
}