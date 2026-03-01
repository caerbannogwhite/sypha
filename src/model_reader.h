#ifndef MODEL_READER_H
#define MODEL_READER_H

#include "common.h"

class SyphaNodeSparse;

SyphaStatus model_reader_read_scp_file_sparse_coo(SyphaNodeSparse &node, std::string inputFilePath);
SyphaStatus model_reader_read_scp_file_sparse_csr(SyphaNodeSparse &node, std::string inputFilePath);

#endif // MODEL_READER_H
