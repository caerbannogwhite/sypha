#ifndef SYPHA_NODE_H
#define SYPHA_NODE_H

#include "common.h"
#include "sypha_environment.h"

class SyphaNode
{
private:
    bool sparse;
    int numCols;
    int numRows;
    int numNonZero;
    double objectiveValue;
    double *h_MatDns;
    double *h_ObjDns;
    double *h_RhsDns;
    double *d_MatDns;
    double *d_ObjDns;
    double *d_RhsDns;
    SyphaEnvironment env;

public:
    SyphaNode(SyphaEnvironment &env);

    bool isSparse();
    int getNumCols();
    int getNumRows();
    int getNumNonZero();
    double getObjectiveValue();
    SyphaStatus solve();
    SyphaStatus importModel();
    SyphaStatus copyModelOnDevice();
    SyphaStatus convert2MySimplexForm();

    friend SyphaStatus model_reader_read_scp_file_dense(SyphaNode &node, string inputFilePath);
    friend SyphaStatus model_reader_read_scp_file_sparse(SyphaNode &node, string inputFilePath);
};

#endif // SYPHA_NODE_H
