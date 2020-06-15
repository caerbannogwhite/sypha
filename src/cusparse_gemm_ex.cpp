// *** spgemm_example.c ***
// How to compile (assume CUDA is installed at /usr/local/cuda/)
//   nvcc spgemm_example.c -o spgemm_example -L/usr/local/cuda/lib64 -lcusparse - lcudart
// or, for C compiler
//   cc -I/usr/local/cuda/include -c spgemm_example.c -o spgemm_example.o -std = c99
//   nvcc -lcusparse -lcudart spgemm_example.o -o spgemm_example
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }
#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }
    int
    main()
{
    // Host problem definition
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_num_nnz = 9;
    const int B_num_rows = 4;
    const int B_num_cols = 4;
    const int B_num_nnz = 9;
    int hA_csrOffsets[] = {0, 3, 4, 7, 9};
    int hA_columns[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    float hA_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         6.0f, 7.0f, 8.0f, 9.0f};
    int hB_csrOffsets[] = {0, 2, 4, 7, 8};
    int hB_columns[] = {0, 3, 1, 3, 0, 1, 2, 1};
    float hB_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         6.0f, 7.0f, 8.0f};
    int hC_csrOffsets[] = {0, 4, 6, 10, 12};
    int hC_columns[] = {0, 1, 2, 3, 1, 3, 0, 1, 2, 3, 1, 3};
    float hC_values[] = {11.0f, 36.0f, 14.0f, 2.0f, 12.0f,
                         16.0f, 35.0f, 92.0f, 42.0f, 10.0f,
                         96.0f, 32.0f};
    const int C_num_nnz = 12;
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B
    int *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
        *dC_csrOffsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    // allocate A
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_num_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_num_nnz * sizeof(float)))
    // allocate B
    CHECK_CUDA(cudaMalloc((void **)&dB_csrOffsets,
                          (B_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dB_columns, B_num_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dB_values, B_num_nnz * sizeof(float)))
    // allocate C offsets
    CHECK_CUDA(cudaMalloc((void **)&dC_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    // copy A
    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values,
                          A_num_nnz * sizeof(float), cudaMemcpyHostToDevice))
    // copy B
    CHECK_CUDA(cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                          (B_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB_columns, hB_columns, B_num_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB_values, hB_values,
                          B_num_nnz * sizeof(float), cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_num_nnz,
                                     dB_csrOffsets, dB_columns, dB_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                     NULL, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))


    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void **)&dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step

    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);
    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);
    cudaMalloc((void **)&dBuffer2, bufferSize2);
    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);

    


    
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);
    // allocate matrix C
    cudaMalloc((void **)&dC_columns, C_num_nnz1 * sizeof(int));
    cudaMalloc((void **)&dC_values, C_num_nnz1 * sizeof(float));
    // update matC with the new pointers
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);
    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    int hC_csrOffsets_tmp[A_num_rows + 1];
    int hC_columns_tmp[C_num_nnz];
    float hC_values_tmp[C_num_nnz];
    CHECK_CUDA(cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hC_columns_tmp, dC_columns,
                          C_num_nnz * sizeof(int),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hC_values_tmp, dC_values,
                          C_num_nnz * sizeof(float),
                          cudaMemcpyDeviceToHost))
    int correct = 1;
    for (int i = 0; i < A_num_rows + 1; i++)
    {
        if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i])
        {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < C_num_nnz; i++)
    {
        if (hC_columns_tmp[i] != hC_columns[i] ||
            hC_values_tmp[i] != hC_values[i])
        {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("spgemm_example test PASSED\n");
    else
    {
        printf("spgemm_example test FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dB_csrOffsets))
    CHECK_CUDA(cudaFree(dB_columns))
    CHECK_CUDA(cudaFree(dB_values))
    CHECK_CUDA(cudaFree(dC_csrOffsets))
    CHECK_CUDA(cudaFree(dC_columns))
    CHECK_CUDA(cudaFree(dC_values))
    return EXIT_SUCCESS;
}