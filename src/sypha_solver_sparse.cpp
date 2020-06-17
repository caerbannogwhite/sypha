
#include "sypha_solver_sparse.h"

SyphaStatus solver_sparse_merhrotra(SyphaNodeSparse &node)
{
    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    double mu = 0.0;
    void *d_buffer = NULL;

    ///////////////////             GET TRANSPOSED MATRIX
    checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransOffs, sizeof(int) * (node.ncols + 1)));
    checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransInds, sizeof(int) * node.nnz));
    checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransVals, sizeof(double) * node.nnz));

    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                                  node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                                  node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
                                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                                  &bufferSize));

    checkCudaErrors(cudaMalloc((void **)&d_buffer, bufferSize));

    checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                       node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                       node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
                                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                       CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                       d_buffer));

    checkCudaErrors(cusparseCreateCsr(&node.matTransDescr, node.ncols, node.nrows, node.nnz,
                      //node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
                      node.d_csrMatTransOffs, node.d_csrMatTransInds, node.d_csrMatTransVals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));


    ///////////////////             GET STARTING POINT

    solver_sparse_merhrotra_init(node);

    while ((iterations < node.env->MERHROTRA_MAX_ITER) && (mu > node.env->MERHROTRA_MU_TOL))
    {

        ++iterations;
    }


    ///////////////////             RELEASE RESOURCES
    checkCudaErrors(cusparseDestroySpMat(node.matTransDescr));

    checkCudaErrors(cudaFree(node.d_csrMatTransInds));
    checkCudaErrors(cudaFree(node.d_csrMatTransOffs));
    checkCudaErrors(cudaFree(node.d_csrMatTransVals));

    checkCudaErrors(cudaFree(d_buffer));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_merhrotra_init(SyphaNodeSparse &node)
{
    int64_t AAT_nrows = node.nrows, AAT_ncols = node.nrows, AAT_nnz = 0;
    double alpha = 1.0;
    double beta = 0.0;

    int *AAT_inds = NULL, *AAT_offs = NULL;
    double *AAT_vals = NULL;

    void *d_buffer1 = NULL, *d_buffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    cusparseSpMatDescr_t AAT_descr;
    cusparseSpGEMMDescr_t spgemmDescr;


    ///////////////////             COMPUTE STARTING COORDINATES X AND S

    // AAT matrix for geMM
    checkCudaErrors(cusparseCreateCsr(&AAT_descr, AAT_nrows, AAT_ncols, AAT_nnz,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // SpGEMM Computation
    checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

    // ask bufferSize1 bytes for external memory
    checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, node.matDescr, node.matTransDescr,
                                  &beta, AAT_descr,
                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDescr, &bufferSize1, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, bufferSize1));

    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, node.matDescr, node.matTransDescr,
                                  &beta, AAT_descr,
                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDescr, &bufferSize1, d_buffer1));

    // ask bufferSize2 bytes for external memory
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, node.matDescr, node.matTransDescr,
                           &beta, AAT_descr,
                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDescr, &bufferSize2, NULL));

    checkCudaErrors(cudaMalloc((void **)&d_buffer2, bufferSize2));

    // compute the intermediate product of A * B
    checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, node.matDescr, node.matTransDescr,
                           &beta, AAT_descr,
                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDescr, &bufferSize2, d_buffer2));

    // get matrix C non-zero entries C_num_nnz1
    //cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);
    cusparseSpMatGetSize(AAT_descr, &AAT_nrows, &AAT_ncols, &AAT_nnz);

    // allocate matrix AAT
    checkCudaErrors(cudaMalloc((void **)&AAT_offs, sizeof(int) * (AAT_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&AAT_inds, sizeof(int) * AAT_nnz));
    checkCudaErrors(cudaMalloc((void **)&AAT_vals, sizeof(double) * AAT_nnz));

    // update AAT with the new pointers
    checkCudaErrors(cusparseCsrSetPointers(AAT_descr, AAT_offs, AAT_inds, AAT_vals));
    
    // copy the final products to the matrix C
    checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, node.matDescr, node.matTransDescr,
                        &beta, AAT_descr,
                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

    checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));



    int64_t r, c, n;
    std::cout << "buff 1: " << bufferSize1 << ", buff 2: " << bufferSize2 << std::endl;

    cusparseSpMatGetSize(node.matDescr, &r, &c, &n);
    std::cout << "\nMat" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;
    cusparseSpMatGetSize(node.matTransDescr, &r, &c, &n);
    std::cout << "\nTrans" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;
    cusparseSpMatGetSize(AAT_descr, &r, &c, &n);
    std::cout << "\nAAT" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;

    
    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));

    checkCudaErrors(cudaFree(AAT_inds));
    checkCudaErrors(cudaFree(AAT_offs));
    checkCudaErrors(cudaFree(AAT_vals));

    return CODE_SUCCESFULL;
}

// #define CHECK_CUDA(func)                                               \
//     {                                                                  \
//         cudaError_t status = (func);                                   \
//         if (status != cudaSuccess)                                     \
//         {                                                              \
//             printf("CUDA API failed at line %d with error: %s (%d)\n", \
//                    __LINE__, cudaGetErrorString(status), status);      \
//             return CODE_ERROR;                                       \
//         }                                                              \
//     }

// #define CHECK_CUSPARSE(func)                                               \
//     {                                                                      \
//         cusparseStatus_t status = (func);                                  \
//         if (status != CUSPARSE_STATUS_SUCCESS)                             \
//         {                                                                  \
//             printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
//                    __LINE__, cusparseGetErrorString(status), status);      \
//             return CODE_ERROR;                                           \
//         }                                                                  \
//     }

// SyphaStatus solver_sparse_merhrotra_init(SyphaNodeSparse &node)
// {
//     // Host problem definition
//     const int B_nrows = 5;
//     const int B_ncols = 2;
//     const int B_nnz = 5;

//     int hB_offs[] = {0, 1, 2, 3, 4, 5};
//     int hB_inds[] = {0, 1, 0, 0, 1};
//     double hB_vals[] = {1, 1, 1, -1, -1};

//     int hC_csrOffsets[] = {0, 1, 2};
//     int hC_columns[] = {0, 1};
//     double hC_values[] = {3, 2};

//     const int C_num_nnz = 2;
//     double alpha = 1.0f;
//     double beta = 0.0f;
//     cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_64F;

//     //--------------------------------------------------------------------------
//     // Device memory management: Allocate and copy A, B
//     int *dB_csrOffsets, *dB_columns;
//     int *dC_csrOffsets, *dC_columns;
//     double *dB_values;
//     double *dC_values;

//     // allocate B
//     CHECK_CUDA(cudaMalloc((void **)&dB_csrOffsets, (B_nrows + 1) * sizeof(int)))
//     CHECK_CUDA(cudaMalloc((void **)&dB_columns, B_nnz * sizeof(int)))
//     CHECK_CUDA(cudaMalloc((void **)&dB_values, B_nnz * sizeof(double)))
//     // allocate C offsets
//     CHECK_CUDA(cudaMalloc((void **)&dC_csrOffsets, (node.nrows + 1) * sizeof(int)))
//     // copy B
//     CHECK_CUDA(cudaMemcpy(dB_csrOffsets, hB_offs,
//                           (B_nrows + 1) * sizeof(int),
//                           cudaMemcpyHostToDevice))
//     CHECK_CUDA(cudaMemcpy(dB_columns, hB_inds, B_nnz * sizeof(int),
//                           cudaMemcpyHostToDevice))
//     CHECK_CUDA(cudaMemcpy(dB_values, hB_vals,
//                           B_nnz * sizeof(double), cudaMemcpyHostToDevice))
//     //--------------------------------------------------------------------------
//     // CUSPARSE APIs
//     cusparseHandle_t handle = NULL;
//     cusparseSpMatDescr_t matB, matC;
//     void *dBuffer1 = NULL, *dBuffer2 = NULL;
//     size_t bufferSize1 = 0, bufferSize2 = 0;
//     CHECK_CUSPARSE(cusparseCreate(&handle))

//     CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_nrows, B_ncols, B_nnz,
//                                      //dB_csrOffsets, dB_columns, dB_values,
//                                      dB_csrOffsets, node.d_csrMatTransInds, node.d_csrMatTransVals,
//                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_BASE_ZERO, computeType))
//     CHECK_CUSPARSE(cusparseCreateCsr(&matC, node.nrows, node.nrows, 0,
//                                      NULL, NULL, NULL,
//                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_BASE_ZERO, computeType))

//     //--------------------------------------------------------------------------
//     // SpGEMM Computation
//     cusparseSpGEMMDescr_t spgemmDesc;
//     CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
//     // ask bufferSize1 bytes for external memory
//     cusparseSpGEMM_workEstimation(handle, opA, opB,
//                                   &alpha, node.matDescr, node.matTransDescr, &beta, matC,
//                                   computeType, CUSPARSE_SPGEMM_DEFAULT,
//                                   spgemmDesc, &bufferSize1, NULL);

//     cudaMalloc((void **)&dBuffer1, bufferSize1);
//     // inspect the matrices A and B to understand the memory requiremnent for
//     // the next step

//     cusparseSpGEMM_workEstimation(handle, opA, opB,
//                                   &alpha, node.matDescr, node.matTransDescr, &beta, matC,
//                                   computeType, CUSPARSE_SPGEMM_DEFAULT,
//                                   spgemmDesc, &bufferSize1, dBuffer1);
//     // ask bufferSize2 bytes for external memory
//     cusparseSpGEMM_compute(handle, opA, opB,
//                            &alpha, node.matDescr, node.matTransDescr, &beta, matC,
//                            computeType, CUSPARSE_SPGEMM_DEFAULT,
//                            spgemmDesc, &bufferSize2, NULL);
//     cudaMalloc((void **)&dBuffer2, bufferSize2);
//     // compute the intermediate product of A * B
//     cusparseSpGEMM_compute(handle, opA, opB,
//                            &alpha, node.matDescr, node.matTransDescr, &beta, matC,
//                            computeType, CUSPARSE_SPGEMM_DEFAULT,
//                            spgemmDesc, &bufferSize2, dBuffer2);

//     // get matrix C non-zero entries C_num_nnz1
//     int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
//     cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);
//     // allocate matrix C
//     cudaMalloc((void **)&dC_columns, C_num_nnz1 * sizeof(int));
//     cudaMalloc((void **)&dC_values, C_num_nnz1 * sizeof(double));
//     // update matC with the new pointers
//     cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);
//     // copy the final products to the matrix C
//     cusparseSpGEMM_copy(handle, opA, opB,
//                         &alpha, node.matDescr, matB, &beta, matC,
//                         computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

//     int64_t r, c, n;

//     cusparseSpMatGetSize(node.matDescr, &r, &c, &n);
//     std::cout << "\nMat" << std::endl;
//     std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;
//     cusparseSpMatGetSize(node.matTransDescr, &r, &c, &n);
//     std::cout << "\nTrans" << std::endl;
//     std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;
//     cusparseSpMatGetSize(matC, &r, &c, &n);
//     std::cout << "\nAAT" << std::endl;
//     std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;

//     // destroy matrix/vector descriptors
//     CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
//     //CHECK_CUSPARSE(cusparseDestroySpMat(matB))
//     CHECK_CUSPARSE(cusparseDestroySpMat(matC))
//     CHECK_CUSPARSE(cusparseDestroy(handle))


//     //--------------------------------------------------------------------------
//     // device result check
//     int hC_csrOffsets_tmp[node.nrows + 1];
//     int hC_columns_tmp[C_num_nnz];
//     double hC_values_tmp[C_num_nnz];
//     CHECK_CUDA(cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
//                           (node.nrows + 1) * sizeof(int),
//                           cudaMemcpyDeviceToHost))
//     CHECK_CUDA(cudaMemcpy(hC_columns_tmp, dC_columns,
//                           C_num_nnz * sizeof(int),
//                           cudaMemcpyDeviceToHost))
//     CHECK_CUDA(cudaMemcpy(hC_values_tmp, dC_values,
//                           C_num_nnz * sizeof(double),
//                           cudaMemcpyDeviceToHost))
//     int correct = 1;
//     for (int i = 0; i < node.nrows + 1; i++)
//     {
//         if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i])
//         {
//             correct = 0;
//             break;
//         }
//     }

//     printf("\nBuff size 1: %lu\nBuff size 2: %lu\n", bufferSize1, bufferSize2);
//     printf("C values:\n");
//     for (int i = 0; i < C_num_nnz; i++)
//     {
//         if (hC_columns_tmp[i] != hC_columns[i] ||
//             hC_values_tmp[i] != hC_values[i])
//         {
//             correct = 0;
//             break;
//         }
//         printf("%.1f ", hC_values_tmp[i]);
//     }
//     printf("\n");

//     if (correct)
//         printf("spgemm_example test PASSED\n");
//     else
//     {
//         printf("spgemm_example test FAILED: wrong result\n");
//         return CODE_ERROR;
//     }
//     //--------------------------------------------------------------------------
//     // device memory deallocation
//     CHECK_CUDA(cudaFree(dBuffer1))
//     CHECK_CUDA(cudaFree(dBuffer2))
//     CHECK_CUDA(cudaFree(dC_csrOffsets))
//     CHECK_CUDA(cudaFree(dC_columns))
//     CHECK_CUDA(cudaFree(dC_values))

    
//     return CODE_SUCCESFULL;
// }