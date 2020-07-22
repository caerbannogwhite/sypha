
#include "sypha_solver_sparse.h"

SyphaStatus solver_sparse_mehrotra(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    size_t bufferSize = 0;
    size_t currBufferSize = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = NULL;
    double *d_bufferS = NULL;
    double *d_buffer = NULL;
    char message[1024];

    cusparseMatDescr_t A_descr;

    ///////////////////             GET TRANSPOSED MATRIX
    
    // checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransOffs, sizeof(int) * (node.ncols + 1)));
    // checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransInds, sizeof(int) * node.nnz));
    // checkCudaErrors(cudaMalloc((void **)&node.d_csrMatTransVals, sizeof(double) * node.nnz));

    // checkCudaErrors(cudaDeviceSynchronize());

    // checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
    //                                               node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
    //                                               node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
    //                                               CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
    //                                               CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
    //                                               &bufferSize));

    // checkCudaErrors(cudaMalloc((void **)&d_buffer, bufferSize));

    // checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
    //                                    node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
    //                                    node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
    //                                    CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
    //                                    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
    //                                    d_buffer));

    // checkCudaErrors(cusparseCreateCsr(&node.matTransDescr, node.ncols, node.nrows, node.nnz,
    //                                   //node.d_csrMatTransVals, node.d_csrMatTransOffs, node.d_csrMatTransInds,
    //                                   node.d_csrMatTransOffs, node.d_csrMatTransInds, node.d_csrMatTransVals,
    //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.h_x = (double *)malloc(sizeof(double) * node.ncols);
    node.h_y = (double *)malloc(sizeof(double) * node.nrows);
    node.h_s = (double *)malloc(sizeof(double) * node.ncols);

    node.timeStartSolStart = node.env->timer();
    solver_sparse_mehrotra_init_gsl(node);
    node.timeStartSolEnd = node.env->timer();

    ///////////////////             SET BIG MATRIX ON HOST
    //
    // On each step we solve this linear system twice:
    //
    //      O | A' | I    x    -rc
    //      --|----|---   -    ---
    //      A | O  | O  * y  = -rb
    //      --|----|---   -    ---
    //      S | O  | X    s    -rxs
    //
    // Where A is the model matrix (standard form), I is the n*n identity
    // matrix, S is the n*n s diagonal matrix, X is the n*n diagonal matrix.
    // Total number of non-zero elements is A.nnz * 2 + n * 3

    node.timePreSolStart = node.env->timer();

    int A_nrows = node.ncols * 2 + node.nrows;
    int A_ncols = A_nrows;
    int A_nnz = node.nnz * 2 + node.ncols * 3;
    
    int *h_csrAInds = NULL;
    int *h_csrAOffs = NULL;
    double *h_csrAVals = NULL;

    int *d_csrAInds = NULL;
    int *d_csrAOffs = NULL;
    double *d_csrAVals = NULL;

    double *d_rhs = NULL;
    double *d_sol = NULL;
    double *d_prevSol = NULL;

    h_csrAInds = (int *)calloc(sizeof(int), A_nnz);
    h_csrAOffs = (int *)calloc(sizeof(int), (A_nrows + 1));
    h_csrAVals = (double *)calloc(sizeof(double), A_nnz);

    sprintf(message, "Initialising matrix: %d rows, %d columns, %d non zeros", A_nrows, A_ncols, A_nnz);
    node.env->logger(message, "INFO", 17);

    // Instantiate the first group of n rows: O | A' | I
    bool found = false;
    int off = 0, rowCnt = 0;

    h_csrAOffs[0] = 0;
    for (j = 0; j < node.ncols; ++j)
    {
        rowCnt = 0;
        for (i = 0; i < node.nrows; ++i)
        {
            found = false;
            for (k = node.h_csrMatOffs->data()[i]; k < node.h_csrMatOffs->data()[i+1]; ++k)
            {
                if (node.h_csrMatInds->data()[k] == j)
                {
                    found = true;
                    break;
                }
            }

            if (found)
            {
                h_csrAInds[off] = node.ncols + i;
                h_csrAVals[off] = node.h_csrMatVals->data()[k];
                ++rowCnt;
                ++off;
            }
        }

        // append the I matrix element for the current row
        h_csrAInds[off] = node.ncols + node.nrows + j;
        h_csrAVals[off] = 1.0;
        ++rowCnt;
        ++off;

        h_csrAOffs[j + 1] = h_csrAOffs[j] + rowCnt;
    }

    // Instantiate the second group of m rows: A | O | O
    for (i = 0; i < node.nrows; ++i)
    {
        h_csrAOffs[node.ncols + i + 1] = h_csrAOffs[node.ncols + i] + (node.h_csrMatOffs->data()[i + 1] - node.h_csrMatOffs->data()[i]);
    }
    memcpy(&h_csrAInds[off], node.h_csrMatInds->data(), sizeof(int) * node.nnz);
    memcpy(&h_csrAVals[off], node.h_csrMatVals->data(), sizeof(double) * node.nnz);
    off += node.nnz;

    // Instantiate the third group of n rows: S | O | X
    for (j = 0; j < node.ncols; ++j)
    {
        // s
        h_csrAInds[off] = j;
        h_csrAVals[off] = node.h_s[j];
        ++off;

        // x
        h_csrAInds[off] = node.ncols + node.nrows + j;
        h_csrAVals[off] = node.h_x[j];
        ++off;

        h_csrAOffs[node.ncols + node.nrows + j + 1] = h_csrAOffs[node.ncols + node.nrows + j] + 2;
    }

    checkCudaErrors(cudaMalloc((void **)&d_csrAInds, sizeof(int) * A_nnz));
    checkCudaErrors(cudaMalloc((void **)&d_csrAOffs, sizeof(int) * (A_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrAVals, sizeof(double) * A_nnz));

    checkCudaErrors(cudaMemcpy(d_csrAInds, h_csrAInds, sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAOffs, h_csrAOffs, sizeof(int) * (A_nrows + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrAVals, h_csrAVals, sizeof(double) * A_nnz, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cusparseCreateMatDescr(&A_descr));
    checkCudaErrors(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));
    
    ///////////////////             TEST
    // double *d_ADn = NULL;
    // checkCudaErrors(cudaMalloc((void **)&d_ADn, sizeof(double) * A_nrows * A_ncols));

    // checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, A_nrows, A_ncols,
    //                                    A_descr, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
    //                                    d_csrAVals, d_csrAOffs, d_csrAInds,
    //                                    d_ADn, A_nrows));

    // utils_printDmat(A_nrows, A_ncols, A_nrows, d_ADn, true);
    // checkCudaErrors(cudaFree(d_ADn));

    // printf("OFFS:\n");
    // utils_printIvec(A_nrows+1, d_csrAOffs, true);
    // printf("INDS:\n");
    // utils_printIvec(A_nnz, d_csrAInds, true);
    // printf("VALS:\n");
    // utils_printDvec(A_nnz, d_csrAVals, true);
    ///////////////////             END TEST

    free(h_csrAInds);
    free(h_csrAOffs);
    free(h_csrAVals);

    ///////////////////             INITIALISE RHS
    
    node.env->logger("Initialise right-hand-side", "INFO", 17);
    checkCudaErrors(cudaMalloc((void **)&d_rhs, sizeof(double) * A_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_sol, sizeof(double) * A_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_prevSol, sizeof(double) * A_nrows));

    // put x, y, s on device sol as [x, y, s]
    double *d_x = d_prevSol;
    double *d_y = &d_prevSol[node.ncols];
    double *d_s = &d_prevSol[node.ncols + node.nrows];

    double *d_deltaX = d_sol;
    double *d_deltaY = &d_sol[node.ncols];
    double *d_deltaS = &d_sol[node.ncols + node.nrows];

    checkCudaErrors(cudaMemcpy(d_x, node.h_x, sizeof(double) * node.ncols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, node.h_y, sizeof(double) * node.nrows, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_s, node.h_s, sizeof(double) * node.ncols, cudaMemcpyHostToDevice));

    // put OBJ and S on device rhs
    double *d_resC = d_rhs;
    double *d_resB = &d_rhs[node.ncols];
    double *d_resXS = &d_rhs[node.ncols + node.nrows];

    checkCudaErrors(cudaMemcpy(d_resC, node.d_ObjDns, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_resB, node.d_RhsDns, sizeof(double) * node.nrows, cudaMemcpyDeviceToDevice));

    // Residuals
    // resB, resC equation 14.7, page 395(414)Numerical Optimization
    // resC = -mat' * y + (obj - s)
    // resB = -mat  * x + rhs

    cusparseDnVecDescr_t vecX, vecY, vecResC, vecResB;

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols, d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows, d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResC, (int64_t)node.ncols, d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, (int64_t)node.nrows, d_resB, CUDA_R_64F));

    alpha = -1.0;
    checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                &alpha, d_s, 1, d_resC, 1));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, vecY,
                                            &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    // buffer size for other needs
    currBufferSize = (size_t)(sizeof(double) * node.ncols * 2);
    currBufferSize = currBufferSize > bufferSize ? currBufferSize : bufferSize;
    checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));

    checkCudaErrors(cusparseSpMV(node.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, vecY,
                                 &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    alpha = -1.0;
    beta = 1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, node.matDescr, vecX,
                                            &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, node.matDescr, vecX,
                                 &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 (size_t *)d_buffer));

    ///////////////////             CALCULATE MU
    // duality measure, defined at page 395(414) Numerical Optimization
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
    mu /= node.ncols;

    node.timePreSolEnd = node.env->timer();

    ///////////////////             MAIN LOOP

    node.env->logger("Starting Mehrotra proceduce", "INFO", 17);
    node.timeSolverStart = node.env->timer();
    while ((iterations < node.env->MEHROTRA_MAX_ITER) && (mu > node.env->MEHROTRA_MU_TOL))
    {


        // x, s multiplication and res XS update: to improve
        //elem_min_mult_hybr(d_x, d_s, d_resXS, node.ncols);
        elem_min_mult_dev(d_x, d_s, d_resXS, node.ncols);
        
        checkCudaErrors(cusolverSpDcsrlsvqr(node.cusolverSpHandle,
                                            A_nrows, A_nnz, A_descr,
                                            d_csrAVals, d_csrAOffs, d_csrAInds,
                                            d_rhs,
                                            node.env->MEHROTRA_CHOL_TOL, reorder,
                                            d_sol, &singularity));

        ///////////////             TEST
        /*printf("\n%4d) AFTER AFFINE SYSTEM\n", iterations);
        double *d_ADn = NULL;
        checkCudaErrors(cudaMalloc((void **)&d_ADn, sizeof(double) * A_nrows * A_ncols));

        checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, A_nrows, A_ncols,
                                           A_descr, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
                                           d_csrAVals, d_csrAOffs, d_csrAInds,
                                           d_ADn, A_nrows));

        utils_printDmat(A_nrows, A_ncols, A_nrows, d_ADn, true, true);
        checkCudaErrors(cudaFree(d_ADn));

        printf("sol:\n");
        utils_printDvec(node.ncols * 2 + node.nrows, d_sol, true);
        printf("rhs:\n");
        utils_printDvec(node.ncols * 2 + node.nrows, d_rhs, true);*/
        ///////////////             END TEST

        // affine step length, definition 14.32 at page 408(427)
        // alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        // alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])

        // finding alphaMaxPrim and alphaMaxDual: to improve
        find_alpha_max(&alphaMaxPrim, &alphaMaxDual,
                       d_x, d_deltaX, d_s, d_deltaS, node.ncols);

        alphaPrim = gsl_min(1.0, alphaMaxPrim);
        alphaDual = gsl_min(1.0, alphaMaxDual);

        // mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s + alpha_aff_d * delta_s_aff) / float(n)
        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system
        // the dimension of the buffer is guaranteed to be >= 2 * ncols
        d_bufferX = d_buffer;
        d_bufferS = &d_buffer[node.ncols];
        checkCudaErrors(cudaMemcpyAsync(d_bufferX, d_x, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
        checkCudaErrors(cudaMemcpyAsync(d_bufferS, d_s, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaPrim, d_deltaX, 1, d_bufferX, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaDual, d_deltaS, 1, d_bufferS, 1));

        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_bufferX, 1, d_bufferS, 1, &muAff));
        muAff /= node.ncols;

        // corrector step or centering parameter
        sigma = gsl_pow_3(muAff / mu);

        ///////////////             TEST
        // printf("\n\n%4d) PRE CORRECTION SYSTEM\n", iterations);
        // printf("sigma: %lf, muAff: %lf\n", sigma, muAff);
        // printf("d buff X:\n");
        // utils_printDvec(node.ncols, d_bufferX, true);
        // printf("d buff S:\n");
        // utils_printDvec(node.ncols, d_bufferS, true);
        ///////////////             END TEST

        // x, s multiplication and res XS update: to improve
        for (j = 0; j < node.ncols; ++j)
        {
            checkCudaErrors(cudaMemcpyAsync(&alpha, &d_deltaX[j], sizeof(double), cudaMemcpyDeviceToHost, node.cudaStream));
            checkCudaErrors(cudaMemcpyAsync(&beta, &d_deltaS[j], sizeof(double), cudaMemcpyDeviceToHost, node.cudaStream));
            alpha = -(alpha * beta) + sigma * mu;
            checkCudaErrors(cudaMemcpyAsync(&d_bufferX[j], &alpha, sizeof(double), cudaMemcpyHostToDevice, node.cudaStream));
        }

        alpha = 1.0;
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alpha, d_bufferX, 1, d_resXS, 1));


        checkCudaErrors(cusolverSpDcsrlsvqr(node.cusolverSpHandle,
                                            A_nrows, A_nnz, A_descr,
                                            d_csrAVals, d_csrAOffs, d_csrAInds,
                                            d_rhs,
                                            node.env->MEHROTRA_CHOL_TOL, reorder,
                                            d_sol, &singularity));
                                            
        ///////////////             TEST
        // printf("\n%4d) AFTER CORRECTION SYSTEM\n", iterations);
        // printf("sol:\n");
        // utils_printDvec(node.ncols * 2 + node.nrows, d_sol, true);
        // printf("rhs:\n");
        // utils_printDvec(node.ncols * 2 + node.nrows, d_rhs, true);
        ///////////////             END TEST

        // finding alphaMaxPrim and alphaMaxDual: to improve
        // finding alphaMaxPrim and alphaMaxDual: to improve
        find_alpha_max(&alphaMaxPrim, &alphaMaxDual,
                       d_x, d_deltaX, d_s, d_deltaS, node.ncols);

        alphaPrim = gsl_min(1.0, node.env->MEHROTRA_ETA * alphaMaxPrim);
        alphaDual = gsl_min(1.0, node.env->MEHROTRA_ETA * alphaMaxDual);

        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system 

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaPrim, d_deltaX, 1, d_x, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.nrows,
                                    &alphaDual, d_deltaY, 1, d_y, 1));
        
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaDual, d_deltaS, 1, d_s, 1));

        ///////////////             UPDATE

        alpha = -(alphaDual - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.ncols,
                                    &alpha, d_resC, 1));

        alpha = -(alphaPrim - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.nrows,
                                    &alpha, d_resB, 1));


        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
        mu /= node.ncols;
        
        ///////////////             TEST
        // printf("\n%4d) UPDATE STEP\n", iterations);
        // printf("mu: %8.6lf, al prim: %8.6lf, al max prim: %8.6lf, al dual: %8.6lf, al max dual: %8.6lf\n", mu, alphaPrim, alphaMaxPrim, alphaDual, alphaMaxDual);
        ///////////////             END TEST

        // update x and s on matrix
        off = node.nnz * 2 + node.ncols;
        checkCudaErrors(cublasDcopy(node.cublasHandle, node.ncols, d_s, 1, &d_csrAVals[off], 2));
        checkCudaErrors(cublasDcopy(node.cublasHandle, node.ncols, d_x, 1, &d_csrAVals[off + 1], 2));

        ++iterations;

        ///////////////             TEST
        //printf("\n\nLOOP END\n");
        //printf("al prim: %lf, al dual: %lf, mu: %lf\n", alphaPrim, alphaDual, mu);
        //printf("X:\n");
        //utils_printDvec(node.ncols, d_x, true);
        //printf("Y:\n");
        //utils_printDvec(node.nrows, d_y, true);
        //printf("S:\n");
        //utils_printDvec(node.ncols, d_s, true);
        //printf("delta X:\n");
        //utils_printDvec(node.ncols, d_deltaX, true);
        //printf("delta S:\n");
        //utils_printDvec(node.ncols, d_deltaS, true);
        ///////////////             END TEST
    }

    node.iterations = iterations;
    
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols,
                               d_x, 1, node.d_ObjDns, 1, &node.objvalPrim));

    checkCudaErrors(cublasDdot(node.cublasHandle, node.nrows,
                               d_y, 1, node.d_RhsDns, 1, &node.objvalDual));

    node.env->logger("Mehrotra procedure complete", "INFO", 10);
    node.timeSolverEnd = node.env->timer();

    ///////////////////             RELEASE RESOURCES

    checkCudaErrors(cusparseDestroyMatDescr(A_descr));

    checkCudaErrors(cudaFree(d_csrAInds));
    checkCudaErrors(cudaFree(d_csrAOffs));
    checkCudaErrors(cudaFree(d_csrAVals));

    checkCudaErrors(cudaFree(d_rhs));
    checkCudaErrors(cudaFree(d_sol));
    checkCudaErrors(cudaFree(d_prevSol));

    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecResC));
    checkCudaErrors(cusparseDestroyDnVec(vecResB));

    checkCudaErrors(cusparseDestroySpMat(node.matTransDescr));
    node.matTransDescr = NULL;

    // checkCudaErrors(cudaFree(node.d_csrMatTransInds));
    // checkCudaErrors(cudaFree(node.d_csrMatTransOffs));
    // checkCudaErrors(cudaFree(node.d_csrMatTransVals));

    // node.d_csrMatTransInds = NULL;
    // node.d_csrMatTransOffs = NULL;
    // node.d_csrMatTransVals = NULL;

    if (d_buffer) checkCudaErrors(cudaFree(d_buffer));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = NULL;
    double *d_bufferS = NULL;

    double *d_resC = NULL, *d_resB = NULL, *d_resXS = NULL;
    double *d_tmpA = NULL, *d_tmpB = NULL;
    double *d_x = NULL, *d_y = NULL, *d_s = NULL, *d_invS = NULL;
    double *d_delX = NULL, *d_delY = NULL, *d_delS = NULL;

    char message[1024];

    cusparseDnVecDescr_t vecX, vecY, vecS, vecResC, vecResB;
    cusparseDnVecDescr_t vecDelX, vecDelY, vecDelS, vecTmpA, vecTmpB;
    cusparseSpGEMMDescr_t spgemmDescr;

    cusparseMatDescr_t matDescrGen;
    checkCudaErrors(cusparseCreateMatDescr(&matDescrGen));
    checkCudaErrors(cusparseSetMatType(matDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    // AT
    int64_t AT_nrows = node.ncols, AT_ncols = node.nrows, AT_nnz = node.nnz;
    int *d_AToffs = NULL, *d_ATinds = NULL;
    double *d_ATvals = NULL;
    cusparseSpMatDescr_t AT_descr;

    checkCudaErrors(cudaMalloc((void **)&d_AToffs, sizeof(int) * (AT_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_ATinds, sizeof(int) * AT_nnz));
    checkCudaErrors(cudaMalloc((void **)&d_ATvals, sizeof(double) * AT_nnz));

    checkCudaErrors(cusparseCreateCsr(&AT_descr, AT_nrows, AT_ncols, AT_nnz,
                                      d_AToffs, d_ATinds, d_ATvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // D
    int64_t D_nrows = node.ncols, D_ncols = node.ncols, D_nnz = node.ncols;
    int *d_Doffs = NULL, *d_Dinds = NULL;
    double *d_Dvals = NULL;
    cusparseSpMatDescr_t D_descr;

    checkCudaErrors(cudaMalloc((void **)&d_Doffs, sizeof(int) * (D_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_Dinds, sizeof(int) * D_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_Dvals, sizeof(double) * D_nrows));

    checkCudaErrors(cusparseCreateCsr(&D_descr, D_nrows, D_ncols, D_nnz,
                                      d_Doffs, d_Dinds, d_Dvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // AD
    int64_t AD_nrows = node.nrows, AD_ncols = node.ncols, AD_nnz = 0, AD_currNnz = 0;
    int *d_ADoffs = NULL, *d_ADinds = NULL;
    double *d_ADvals = NULL;
    cusparseSpMatDescr_t AD_descr;

    checkCudaErrors(cudaMalloc((void **)&d_ADoffs, sizeof(int) * (AD_nrows + 1)));

    checkCudaErrors(cusparseCreateCsr(&AD_descr, AD_nrows, AD_ncols, AD_nnz,
                                      d_ADoffs, d_ADinds, d_ADvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // ADA
    int64_t ADA_nrows = node.nrows, ADA_ncols = node.nrows, ADA_nnz = 0, ADA_currNnz = 0;
    int *d_ADAoffs = NULL, *d_ADAinds = NULL;
    double *d_ADAvals = NULL;
    cusparseSpMatDescr_t ADA_descr;

    checkCudaErrors(cudaMalloc((void **)&d_ADAoffs, sizeof(int) * (ADA_nrows + 1)));
    
    checkCudaErrors(cusparseCreateCsr(&ADA_descr, ADA_nrows, ADA_ncols, ADA_nnz,
                                      d_ADAoffs, d_ADAinds, d_ADAvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // BUFFER
    size_t bufferSize1 = 0;
    size_t currBufferSize1 = (size_t)(sizeof(double) * node.ncols * 2);
    double *d_buffer1 = NULL;

    size_t bufferSize2 = 0;
    size_t currBufferSize2 = 0;
    double *d_buffer2 = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));

    ///////////////////             GET TRANSPOSED MATRIX

    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                                  node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                                  d_ATvals, d_AToffs, d_ATinds,
                                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                                  &bufferSize1));
    
    if (bufferSize1 > currBufferSize1)
    {
        currBufferSize1 = bufferSize1;
        if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
        checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
    }

    checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                       node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                       d_ATvals, d_AToffs, d_ATinds,
                                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                       CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                       d_buffer1));

    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.h_x = (double *)malloc(sizeof(double) * node.ncols);
    node.h_y = (double *)malloc(sizeof(double) * node.nrows);
    node.h_s = (double *)malloc(sizeof(double) * node.ncols);

    node.timeStartSolStart = node.env->timer();
    solver_sparse_mehrotra_init_gsl(node);
    node.timeStartSolEnd = node.env->timer();

    ///////////////////             INITIALISE RHS

    node.env->logger("Initialise right-hand-side", "INFO", 17);
    node.timePreSolStart = node.env->timer();

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_s, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMalloc((void **)&d_delX, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_delY, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_delS, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_x, node.h_x, sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_y, node.h_y, sizeof(double) * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_s, node.h_s, sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));

    // put OBJ and S on device rhs
    checkCudaErrors(cudaMalloc((void **)&d_resC, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_resB, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_resXS, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_resC, node.d_ObjDns, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_resB, node.d_RhsDns, sizeof(double) * node.nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

    // Residuals
    // resB, resC equation 14.7, page 395(414)Numerical Optimization
    // resC = AT * Y - (obj - s)
    // resB = A  * X - rhs

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols, d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows, d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecS, (int64_t)node.ncols, d_s, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecDelX, (int64_t)node.ncols, d_delX, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecDelY, (int64_t)node.nrows, d_delY, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecDelS, (int64_t)node.ncols, d_delS, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecResC, (int64_t)node.ncols, d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, (int64_t)node.nrows, d_resB, CUDA_R_64F));

    alpha = -1.0;
    checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                &alpha, d_s, 1, d_resC, 1));

    alpha = 1.0;
    beta = -1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, AT_descr, vecY,
                                            &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize1));

    if (bufferSize1 > currBufferSize1)
    {
        currBufferSize1 = bufferSize1;
        if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
        checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, AT_descr, vecY,
                                 &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer1));

    alpha = 1.0;
    beta = -1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, node.matDescr, vecX,
                                            &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize1));

    if (bufferSize1 > currBufferSize1)
    {
        currBufferSize1 = bufferSize1;
        if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
        checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, node.matDescr, vecX,
                                 &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 (size_t *)d_buffer1));

    ///////////////////             CALCULATE MU
    // duality measure, defined at page 395(414) Numerical Optimization
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
    mu /= node.ncols;

    node.timePreSolEnd = node.env->timer();

    ///////////////////             SET UP D AND INV(S)
    
    int bSize = (node.ncols >> 5) + 1;

    checkCudaErrors(cudaMalloc((void **)&d_invS, sizeof(double) * node.ncols));
    
    cudaDeviceSynchronize();
    range_kernel<<<bSize, 32>>>(d_Doffs, node.ncols + 1);
    range_kernel<<<bSize, 32>>>(d_Dinds, node.ncols);
    
    ///////////////////             MAIN LOOP
    
    node.env->logger("Starting Mehrotra proceduce", "INFO", 17);
    node.timeSolverStart = node.env->timer();
    
    iterations = 0;
    while ((iterations < node.env->MEHROTRA_MAX_ITER) && (mu > node.env->MEHROTRA_MU_TOL))
    {
        // x, s multiplication and res XS update: to improve
        cudaDeviceSynchronize();
        elem_mult_kernel<<<bSize, 32>>>(d_x, d_s, d_resXS, node.ncols);

        elem_inv_kernel<<<bSize, 32>>>(d_s, d_invS, node.ncols);

        cudaDeviceSynchronize();
        elem_mult_kernel<<<bSize, 32>>>(d_x, d_invS, d_Dvals, node.ncols);

        // cudaDeviceSynchronize();
        // printf("%d) START\n", iterations);
        // printf("X\n");
        // utils_printDvec(node.ncols, d_x, true);
        // printf("INV(S)\n");
        // utils_printDvec(node.ncols, d_invS, true);
        // printf("D\n");
        // utils_printDvec(node.ncols, d_Dvals, true);
        // printf("RES B\n");
        // utils_printDvec(node.nrows, d_resB, true);
        // printf("RES C\n");
        // utils_printDvec(node.ncols, d_resC, true);

        ///////////////             COMPUTE AD
        {
            alpha = 1.0;
            beta = 0.0;

            // SpGEMM Computation
            checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

            // ask bufferSize1 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, D_descr,
                                                  &beta, AD_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, NULL));

            if (bufferSize1 > currBufferSize1)
            {
                currBufferSize1 = bufferSize1;
                if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
                checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
            }

            // inspect the matrices D and AT to understand the memory requiremnent for
            // the next step
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, D_descr,
                                                  &beta, AD_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, d_buffer1));

            // ask bufferSize2 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, D_descr,
                                           &beta, AD_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, NULL));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            // compute the intermediate product of A * B
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, D_descr,
                                           &beta, AD_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

            // get matrix DA non-zero entries
            cusparseSpMatGetSize(AD_descr, &AD_nrows, &AD_ncols, &AD_nnz);

            // allocate matrix DA
            if (AD_nnz > AD_currNnz)
            {
                AD_currNnz = AD_nnz;
                if (d_ADinds) checkCudaErrors(cudaFree(d_ADinds));
                if (d_ADvals) checkCudaErrors(cudaFree(d_ADvals));
                checkCudaErrors(cudaMalloc((void **)&d_ADinds, sizeof(int) * AD_currNnz));
                checkCudaErrors(cudaMalloc((void **)&d_ADvals, sizeof(double) * AD_currNnz));
            }

            // update DA with the new pointers
            checkCudaErrors(cusparseCsrSetPointers(AD_descr, d_ADoffs, d_ADinds, d_ADvals));

            // copy the final products to the matrix AAT
            checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, node.matDescr, D_descr,
                                        &beta, AD_descr,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

            checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));
        }

        ///////////////             COMPUTE ADA
        {
            alpha = 1.0;
            beta = 0.0;

            // SpGEMM Computation
            checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

            // ask bufferSize1 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, AD_descr, AT_descr,
                                                  &beta, ADA_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, NULL));

            if (bufferSize1 > currBufferSize1)
            {
                currBufferSize1 = bufferSize1;
                if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
                checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
            }

            // inspect the matrices A and DA to understand the memory requiremnent for
            // the next step
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, AD_descr, AT_descr,
                                                  &beta, ADA_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, d_buffer1));

            // ask bufferSize2 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, AD_descr, AT_descr,
                                           &beta, ADA_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, NULL));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            // compute the intermediate product of A * DA
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, AD_descr, AT_descr,
                                           &beta, ADA_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

            // get matrix ADA non-zero entries
            cusparseSpMatGetSize(ADA_descr, &ADA_nrows, &ADA_ncols, &ADA_nnz);

            // allocate matrix ADA
            if (ADA_nnz > ADA_currNnz)
            {
                ADA_currNnz = ADA_nnz;
                if (d_ADAinds) checkCudaErrors(cudaFree(d_ADAinds));
                if (d_ADAvals) checkCudaErrors(cudaFree(d_ADAvals));
                checkCudaErrors(cudaMalloc((void **)&d_ADAinds, sizeof(int) * ADA_currNnz));
                checkCudaErrors(cudaMalloc((void **)&d_ADAvals, sizeof(double) * ADA_currNnz));
            }

            // update ADA with the new pointers
            checkCudaErrors(cusparseCsrSetPointers(ADA_descr, d_ADAoffs, d_ADAinds, d_ADAvals));

            // copy the final products to the matrix ADA
            checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, AD_descr, AT_descr,
                                        &beta, ADA_descr,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

            checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));
        }

        ///////////////////             TEST
        // double *d_ADn = NULL;
        // checkCudaErrors(cudaMalloc((void **)&d_ADn, sizeof(double) * ADA_nrows * ADA_ncols));

        // cusparseMatDescr_t matGenDescr;
        // checkCudaErrors(cusparseCreateMatDescr(&matGenDescr));
        // checkCudaErrors(cusparseSetMatType(matGenDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
        // checkCudaErrors(cusparseSetMatIndexBase(matGenDescr, CUSPARSE_INDEX_BASE_ZERO));

        // checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, ADA_nrows, ADA_ncols,
        //                                    matGenDescr, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
        //                                    d_ADAvals, d_ADAoffs, d_ADAinds,
        //                                    d_ADn, ADA_nrows));

        // printf("%d) MAT\n", iterations);
        // utils_printDmat(ADA_nrows, ADA_ncols, ADA_nrows, d_ADn, true, false);
        // checkCudaErrors(cudaFree(d_ADn));

        // printf("OFFS:\n");
        // utils_printIvec(A_nrows+1, d_csrAOffs, true);
        // printf("INDS:\n");
        // utils_printIvec(A_nnz, d_csrAInds, true);
        // printf("VALS:\n");
        // utils_printDvec(A_nnz, d_csrAVals, true);
        ///////////////////             END TEST

        ///////////////             COMPUTE TMPA = - AD * resC - resB

        // store TMPA vector on buffer 1, size of buffer 1 is guaranteed to 
        // be >= 2*ncols, copy resB on TMPA
        d_tmpA = d_buffer1;
        d_tmpB = &d_buffer1[node.ncols];
        checkCudaErrors(cusparseCreateDnVec(&vecTmpA, AD_nrows, d_tmpA, CUDA_R_64F));
        checkCudaErrors(cudaMemcpyAsync(d_tmpA, d_resB, sizeof(double) * AD_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        // compute TMPB
        elem_mult_kernel<<<bSize, 32>>>(d_resXS, d_invS, d_tmpB, node.ncols);
        checkCudaErrors(cusparseCreateDnVec(&vecTmpB, AD_ncols, d_tmpB, CUDA_R_64F));

        {    
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AD_descr, vecResC,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AD_descr, vecResC,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));
        }

        ///////////////             COMPUTE TMPA = TMPA + A * TMPB
        {    
            alpha = 1.0;
            beta = 1.0;

            cudaDeviceSynchronize();

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, node.matDescr, vecTmpB,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, node.matDescr, vecTmpB,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));
        }

        ///////////////             COMPUTE DEL_Y = SOLVE(ADA, TMPA)
        {
            checkCudaErrors(cusolverSpDcsrlsvchol(node.cusolverSpHandle,
                                            ADA_nrows, ADA_nnz, matDescrGen,
                                            d_ADAvals, d_ADAoffs, d_ADAinds,
                                            d_tmpA,
                                            node.env->MEHROTRA_CHOL_TOL, reorder,
                                            d_delY, &singularity));
        }

        ///////////////             COMPUTE DEL_S = - AT * DEL_Y - resC
        checkCudaErrors(cudaMemcpyAsync(d_delS, d_resC, sizeof(double) * AT_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        {
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AT_descr, vecDelY,
                        &beta, vecDelS, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AT_descr, vecDelY,
                        &beta, vecDelS, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));   
        }

        ///////////////             COMPUTE DEL_X = -TMP_B - D * DEL_S
        checkCudaErrors(cudaMemcpyAsync(d_delX, d_tmpB, sizeof(double) * D_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        {
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, D_descr, vecDelS,
                        &beta, vecDelX, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, D_descr, vecDelS,
                        &beta, vecDelX, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));   
        }

        // printf("%d) AFFINE SYSTEM\n", iterations);
        // printf("delta X\n");
        // utils_printDvec(node.ncols, d_delX, true);
        // printf("delta Y\n");
        // utils_printDvec(node.nrows, d_delY, true);
        // printf("delta S\n");
        // utils_printDvec(node.ncols, d_delS, true);

        // affine step length, definition 14.32 at page 408(427)
        // alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        // alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])

        // finding alphaMaxPrim and alphaMaxDual: to improve
        find_alpha_max(&alphaMaxPrim, &alphaMaxDual,
                       d_x, d_delX, d_s, d_delS, node.ncols);

        alphaPrim = gsl_min(1.0, alphaMaxPrim);
        alphaDual = gsl_min(1.0, alphaMaxDual);

        // mu_aff = (x + alpha_aff_p * delta_x_aff).dot(s + alpha_aff_d * delta_s_aff) / float(n)
        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system
        // the dimension of the buffer is guaranteed to be >= 2 * ncols
        //d_tmpA = d_buffer1;
        //d_tmpB = &d_buffer1[node.ncols];
        checkCudaErrors(cudaMemcpyAsync(d_tmpA, d_x, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
        checkCudaErrors(cudaMemcpyAsync(d_tmpB, d_s, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols, &alphaPrim, d_delX, 1, d_tmpA, 1));
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols, &alphaDual, d_delS, 1, d_tmpB, 1));

        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_tmpA, 1, d_tmpB, 1, &muAff));
        muAff /= node.ncols;

        // corrector step or centering parameter
        sigma = gsl_pow_3(muAff / mu);

        elem_mult_kernel<<<bSize, 32>>>(d_delX, d_delS, d_tmpA, node.ncols);
        cudaDeviceSynchronize();

        alpha = 1.0;
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols, &alpha, d_tmpA, 1, d_resXS, 1));

        alpha = - sigma * mu;
        scal_sum_kernel<<<bSize, 32>>>(alpha, d_resXS, node.ncols);
        cudaDeviceSynchronize();

        ///////////////             COMPUTE TMPA = - AD * resC - resB

        // store TMPA vector on buffer 1, size of buffer 1 is guaranteed to 
        // be >= 2*ncols, copy resB on TMPA
        checkCudaErrors(cusparseCreateDnVec(&vecTmpA, AD_nrows, d_tmpA, CUDA_R_64F));
        checkCudaErrors(cudaMemcpyAsync(d_tmpA, d_resB, sizeof(double) * AD_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        // compute TMPB
        elem_mult_kernel<<<bSize, 32>>>(d_resXS, d_invS, d_tmpB, node.ncols);
        checkCudaErrors(cusparseCreateDnVec(&vecTmpB, AD_ncols, d_tmpB, CUDA_R_64F));

        {    
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AD_descr, vecResC,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AD_descr, vecResC,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));
        }

        ///////////////             COMPUTE TMPA = TMPA + A * TMPB
        {    
            alpha = 1.0;
            beta = 1.0;

            cudaDeviceSynchronize();

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, node.matDescr, vecTmpB,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, node.matDescr, vecTmpB,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));
        }

        ///////////////             COMPUTE DEL_Y = SOLVE(ADA, TMPA)
        {
            checkCudaErrors(cusolverSpDcsrlsvchol(node.cusolverSpHandle,
                                            ADA_nrows, ADA_nnz, matDescrGen,
                                            d_ADAvals, d_ADAoffs, d_ADAinds,
                                            d_tmpA,
                                            node.env->MEHROTRA_CHOL_TOL, reorder,
                                            d_delY, &singularity));
        }

        ///////////////             COMPUTE DEL_S = - AT * DEL_Y - resC
        checkCudaErrors(cudaMemcpyAsync(d_delS, d_resC, sizeof(double) * AT_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        {
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AT_descr, vecDelY,
                        &beta, vecDelS, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AT_descr, vecDelY,
                        &beta, vecDelS, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));   
        }

        ///////////////             COMPUTE DEL_X = -TMP_B - D * DEL_S
        checkCudaErrors(cudaMemcpyAsync(d_delX, d_tmpB, sizeof(double) * D_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        {
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, D_descr, vecDelS,
                        &beta, vecDelX, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, D_descr, vecDelS,
                        &beta, vecDelX, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));   
        }

        // printf("%d) CORRECTION SYSTEM\n", iterations);
        // printf("delta X\n");
        // utils_printDvec(node.ncols, d_delX, true);
        // printf("delta Y\n");
        // utils_printDvec(node.nrows, d_delY, true);
        // printf("delta S\n");
        // utils_printDvec(node.ncols, d_delS, true);

        // finding alphaMaxPrim and alphaMaxDual: to improve
        find_alpha_max(&alphaMaxPrim, &alphaMaxDual,
                       d_x, d_delX, d_s, d_delS, node.ncols);

        alphaPrim = gsl_min(1.0, node.env->MEHROTRA_ETA * alphaMaxPrim);
        alphaDual = gsl_min(1.0, node.env->MEHROTRA_ETA * alphaMaxDual);

        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system 

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaPrim, d_delX, 1, d_x, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.nrows,
                                    &alphaDual, d_delY, 1, d_y, 1));
        
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaDual, d_delS, 1, d_s, 1));

        ///////////////             UPDATE

        alpha = -(alphaDual - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.ncols,
                                    &alpha, d_resC, 1));

        alpha = -(alphaPrim - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.nrows,
                                    &alpha, d_resB, 1));

        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
        mu /= node.ncols;

        ++iterations;
    }

    node.iterations = iterations;
    
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols,
                               d_x, 1, node.d_ObjDns, 1, &node.objvalPrim));

    checkCudaErrors(cublasDdot(node.cublasHandle, node.nrows,
                               d_y, 1, node.d_RhsDns, 1, &node.objvalDual));

    node.env->logger("Mehrotra procedure complete", "INFO", 10);
    node.timeSolverEnd = node.env->timer();

    ///////////////////             RELEASE MEMORY

    free(node.h_x);
    free(node.h_y);
    free(node.h_s);

    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_s));

    checkCudaErrors(cudaFree(d_delX));
    checkCudaErrors(cudaFree(d_delY));
    checkCudaErrors(cudaFree(d_delS));
    
    checkCudaErrors(cudaFree(d_resC));
    checkCudaErrors(cudaFree(d_resB));
    checkCudaErrors(cudaFree(d_resXS));
    
    checkCudaErrors(cudaFree(d_invS));
    
    checkCudaErrors(cusparseDestroyMatDescr(matDescrGen));

    cusparseDestroySpMat(AT_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroySpMat(AD_descr);
    cusparseDestroySpMat(ADA_descr);

    checkCudaErrors(cudaFree(d_AToffs));
    checkCudaErrors(cudaFree(d_ATinds));
    checkCudaErrors(cudaFree(d_ATvals));
    
    checkCudaErrors(cudaFree(d_Dvals));
    checkCudaErrors(cudaFree(d_Doffs));
    checkCudaErrors(cudaFree(d_Dinds));

    checkCudaErrors(cudaFree(d_ADvals));
    checkCudaErrors(cudaFree(d_ADoffs));
    checkCudaErrors(cudaFree(d_ADinds));

    checkCudaErrors(cudaFree(d_ADAvals));
    checkCudaErrors(cudaFree(d_ADAoffs));
    checkCudaErrors(cudaFree(d_ADAinds));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_3(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int i = 0, j = 0, k = 0, iterations = 0;
    double alpha, beta, alphaPrim, alphaDual, sigma, mu, muAff;
    double alphaMaxPrim, alphaMaxDual;
    double *d_bufferX = NULL;
    double *d_bufferS = NULL;

    double *d_resC = NULL, *d_resB = NULL, *d_resXS = NULL;
    double *d_tmpA = NULL, *d_tmpB = NULL;
    double *d_x = NULL, *d_y = NULL, *d_s = NULL, *d_invS = NULL;
    double *d_delX = NULL, *d_delY = NULL, *d_delS = NULL;

    char message[1024];

    cusparseDnVecDescr_t vecX, vecY, vecS, vecResC, vecResB;
    cusparseDnVecDescr_t vecDelX, vecDelY, vecDelS, vecTmpA, vecTmpB;
    cusparseSpGEMMDescr_t spgemmDescr;

    cusparseMatDescr_t matDescrGen;
    checkCudaErrors(cusparseCreateMatDescr(&matDescrGen));
    checkCudaErrors(cusparseSetMatType(matDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    // AT
    int64_t AT_nrows = node.ncols, AT_ncols = node.nrows, AT_nnz = node.nnz;
    int *d_AToffs = NULL, *d_ATinds = NULL;
    double *d_ATvals = NULL;
    cusparseSpMatDescr_t AT_descr;

    checkCudaErrors(cudaMalloc((void **)&d_AToffs, sizeof(int) * (AT_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_ATinds, sizeof(int) * AT_nnz));
    checkCudaErrors(cudaMalloc((void **)&d_ATvals, sizeof(double) * AT_nnz));

    checkCudaErrors(cusparseCreateCsr(&AT_descr, AT_nrows, AT_ncols, AT_nnz,
                                      d_AToffs, d_ATinds, d_ATvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // D
    int64_t D_nrows = node.ncols, D_ncols = node.ncols, D_nnz = node.ncols;
    int *d_Doffs = NULL, *d_Dinds = NULL;
    double *d_Dvals = NULL;
    cusparseSpMatDescr_t D_descr;

    checkCudaErrors(cudaMalloc((void **)&d_Doffs, sizeof(int) * (D_nrows + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_Dinds, sizeof(int) * D_nrows));
    checkCudaErrors(cudaMalloc((void **)&d_Dvals, sizeof(double) * D_nrows));

    checkCudaErrors(cusparseCreateCsr(&D_descr, D_nrows, D_ncols, D_nnz,
                                      d_Doffs, d_Dinds, d_Dvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // AD
    int64_t AD_nrows = node.nrows, AD_ncols = node.ncols, AD_nnz = 0, AD_currNnz = 0;
    int *d_ADoffs = NULL, *d_ADinds = NULL;
    double *d_ADvals = NULL;
    cusparseSpMatDescr_t AD_descr;

    checkCudaErrors(cudaMalloc((void **)&d_ADoffs, sizeof(int) * (AD_nrows + 1)));

    checkCudaErrors(cusparseCreateCsr(&AD_descr, AD_nrows, AD_ncols, AD_nnz,
                                      d_ADoffs, d_ADinds, d_ADvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // ADA
    int64_t ADA_nrows = node.nrows, ADA_ncols = node.nrows, ADA_nnz = 0, ADA_currNnz = 0;
    int *d_ADAoffs = NULL, *d_ADAinds = NULL;
    double *d_ADAvals = NULL;
    cusparseSpMatDescr_t ADA_descr;

    checkCudaErrors(cudaMalloc((void **)&d_ADAoffs, sizeof(int) * (ADA_nrows + 1)));
    
    checkCudaErrors(cusparseCreateCsr(&ADA_descr, ADA_nrows, ADA_ncols, ADA_nnz,
                                      d_ADAoffs, d_ADAinds, d_ADAvals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // BUFFER
    size_t bufferSize1 = 0;
    size_t currBufferSize1 = (size_t)(sizeof(double) * node.ncols * 2);
    double *d_buffer1 = NULL;

    size_t bufferSize2 = 0;
    size_t currBufferSize2 = 0;
    double *d_buffer2 = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));

    ///////////////////             GET TRANSPOSED MATRIX

    checkCudaErrors(cusparseCsr2cscEx2_bufferSize(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                                  node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                                  d_ATvals, d_AToffs, d_ATinds,
                                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                                  &bufferSize1));
    
    if (bufferSize1 > currBufferSize1)
    {
        currBufferSize1 = bufferSize1;
        if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
        checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
    }

    checkCudaErrors(cusparseCsr2cscEx2(node.cusparseHandle, node.nrows, node.ncols, node.nnz,
                                       node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                       d_ATvals, d_AToffs, d_ATinds,
                                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                       CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                                       d_buffer1));

    ///////////////////             GET STARTING POINT
    // initialise x, y, s
    node.h_x = (double *)malloc(sizeof(double) * node.ncols);
    node.h_y = (double *)malloc(sizeof(double) * node.nrows);
    node.h_s = (double *)malloc(sizeof(double) * node.ncols);

    node.timeStartSolStart = node.env->timer();
    solver_sparse_mehrotra_init_gsl(node);
    node.timeStartSolEnd = node.env->timer();

    ///////////////////             INITIALISE RHS

    node.env->logger("Initialise right-hand-side", "INFO", 17);
    node.timePreSolStart = node.env->timer();

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_y, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_s, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMalloc((void **)&d_delX, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_delY, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_delS, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_x, node.h_x, sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_y, node.h_y, sizeof(double) * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_s, node.h_s, sizeof(double) * node.ncols, cudaMemcpyHostToDevice, node.cudaStream));

    // put OBJ and S on device rhs
    checkCudaErrors(cudaMalloc((void **)&d_resC, sizeof(double) * node.ncols));
    checkCudaErrors(cudaMalloc((void **)&d_resB, sizeof(double) * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_resXS, sizeof(double) * node.ncols));

    checkCudaErrors(cudaMemcpyAsync(d_resC, node.d_ObjDns, sizeof(double) * node.ncols, cudaMemcpyDeviceToDevice, node.cudaStream));
    checkCudaErrors(cudaMemcpyAsync(d_resB, node.d_RhsDns, sizeof(double) * node.nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

    // Residuals
    // resB, resC equation 14.7, page 395(414)Numerical Optimization
    // resC = AT * Y - (obj - s)
    // resB = A  * X - rhs

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols, d_x, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows, d_y, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecS, (int64_t)node.ncols, d_s, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecDelX, (int64_t)node.ncols, d_delX, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecDelY, (int64_t)node.nrows, d_delY, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecDelS, (int64_t)node.ncols, d_delS, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecResC, (int64_t)node.ncols, d_resC, CUDA_R_64F));
    checkCudaErrors(cusparseCreateDnVec(&vecResB, (int64_t)node.nrows, d_resB, CUDA_R_64F));

    alpha = -1.0;
    checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                &alpha, d_s, 1, d_resC, 1));

    alpha = 1.0;
    beta = -1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, AT_descr, vecY,
                                            &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize1));

    if (bufferSize1 > currBufferSize1)
    {
        currBufferSize1 = bufferSize1;
        if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
        checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, AT_descr, vecY,
                                 &beta, vecResC, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer1));

    alpha = 1.0;
    beta = -1.0;
    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, node.matDescr, vecX,
                                            &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize1));

    if (bufferSize1 > currBufferSize1)
    {
        currBufferSize1 = bufferSize1;
        if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
        checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
    }

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, node.matDescr, vecX,
                                 &beta, vecResB, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 (size_t *)d_buffer1));

    ///////////////////             CALCULATE MU
    // duality measure, defined at page 395(414) Numerical Optimization
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
    mu /= node.ncols;

    node.timePreSolEnd = node.env->timer();

    ///////////////////             SET UP D AND INV(S)
    
    int bSize = (node.ncols >> 5) + 1;

    checkCudaErrors(cudaMalloc((void **)&d_invS, sizeof(double) * node.ncols));
    
    cudaDeviceSynchronize();
    range_kernel<<<bSize, 32>>>(d_Doffs, node.ncols + 1);
    range_kernel<<<bSize, 32>>>(d_Dinds, node.ncols);
    
    ///////////////////             MAIN LOOP
    
    node.env->logger("Starting Mehrotra proceduce", "INFO", 17);
    node.timeSolverStart = node.env->timer();
    
    iterations = 0;
    while ((iterations < node.env->MEHROTRA_MAX_ITER) && (mu > node.env->MEHROTRA_MU_TOL))
    {
        // x, s multiplication and res XS update: to improve
        cudaDeviceSynchronize();
        elem_mult_kernel<<<bSize, 32>>>(d_x, d_s, d_resXS, node.ncols);

        elem_inv_kernel<<<bSize, 32>>>(d_s, d_invS, node.ncols);

        cudaDeviceSynchronize();
        elem_mult_kernel<<<bSize, 32>>>(d_x, d_invS, d_Dvals, node.ncols);

        // cudaDeviceSynchronize();
        // printf("%d) START\n", iterations);
        // printf("X\n");
        // utils_printDvec(node.ncols, d_x, true);
        // printf("INV(S)\n");
        // utils_printDvec(node.ncols, d_invS, true);
        // printf("D\n");
        // utils_printDvec(node.ncols, d_Dvals, true);
        // printf("RES B\n");
        // utils_printDvec(node.nrows, d_resB, true);
        // printf("RES C\n");
        // utils_printDvec(node.ncols, d_resC, true);

        ///////////////             COMPUTE AD
        {
            alpha = 1.0;
            beta = 0.0;

            // SpGEMM Computation
            checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

            // ask bufferSize1 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, D_descr,
                                                  &beta, AD_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, NULL));

            if (bufferSize1 > currBufferSize1)
            {
                currBufferSize1 = bufferSize1;
                if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
                checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
            }

            // inspect the matrices D and AT to understand the memory requiremnent for
            // the next step
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, node.matDescr, D_descr,
                                                  &beta, AD_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, d_buffer1));

            // ask bufferSize2 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, D_descr,
                                           &beta, AD_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, NULL));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            // compute the intermediate product of A * B
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, node.matDescr, D_descr,
                                           &beta, AD_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

            // get matrix DA non-zero entries
            cusparseSpMatGetSize(AD_descr, &AD_nrows, &AD_ncols, &AD_nnz);

            // allocate matrix DA
            if (AD_nnz > AD_currNnz)
            {
                AD_currNnz = AD_nnz;
                if (d_ADinds) checkCudaErrors(cudaFree(d_ADinds));
                if (d_ADvals) checkCudaErrors(cudaFree(d_ADvals));
                checkCudaErrors(cudaMalloc((void **)&d_ADinds, sizeof(int) * AD_currNnz));
                checkCudaErrors(cudaMalloc((void **)&d_ADvals, sizeof(double) * AD_currNnz));
            }

            // update DA with the new pointers
            checkCudaErrors(cusparseCsrSetPointers(AD_descr, d_ADoffs, d_ADinds, d_ADvals));

            // copy the final products to the matrix AAT
            checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, node.matDescr, D_descr,
                                        &beta, AD_descr,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

            checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));
        }

        ///////////////             COMPUTE ADA
        {
            alpha = 1.0;
            beta = 0.0;

            // SpGEMM Computation
            checkCudaErrors(cusparseSpGEMM_createDescr(&spgemmDescr));

            // ask bufferSize1 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, AD_descr, AT_descr,
                                                  &beta, ADA_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, NULL));

            if (bufferSize1 > currBufferSize1)
            {
                currBufferSize1 = bufferSize1;
                if (d_buffer1) checkCudaErrors(cudaFree(d_buffer1));
                checkCudaErrors(cudaMalloc((void **)&d_buffer1, currBufferSize1));
            }

            // inspect the matrices A and DA to understand the memory requiremnent for
            // the next step
            checkCudaErrors(cusparseSpGEMM_workEstimation(node.cusparseHandle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, AD_descr, AT_descr,
                                                  &beta, ADA_descr,
                                                  CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDescr, &bufferSize1, d_buffer1));

            // ask bufferSize2 bytes for external memory
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, AD_descr, AT_descr,
                                           &beta, ADA_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, NULL));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            // compute the intermediate product of A * DA
            checkCudaErrors(cusparseSpGEMM_compute(node.cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, AD_descr, AT_descr,
                                           &beta, ADA_descr,
                                           CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDescr, &bufferSize2, d_buffer2));

            // get matrix ADA non-zero entries
            cusparseSpMatGetSize(ADA_descr, &ADA_nrows, &ADA_ncols, &ADA_nnz);

            // allocate matrix ADA
            if (ADA_nnz > ADA_currNnz)
            {
                ADA_currNnz = ADA_nnz;
                if (d_ADAinds) checkCudaErrors(cudaFree(d_ADAinds));
                if (d_ADAvals) checkCudaErrors(cudaFree(d_ADAvals));
                checkCudaErrors(cudaMalloc((void **)&d_ADAinds, sizeof(int) * ADA_currNnz));
                checkCudaErrors(cudaMalloc((void **)&d_ADAvals, sizeof(double) * ADA_currNnz));
            }

            // update ADA with the new pointers
            checkCudaErrors(cusparseCsrSetPointers(ADA_descr, d_ADAoffs, d_ADAinds, d_ADAvals));

            // copy the final products to the matrix ADA
            checkCudaErrors(cusparseSpGEMM_copy(node.cusparseHandle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, AD_descr, AT_descr,
                                        &beta, ADA_descr,
                                        CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

            checkCudaErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));
        }

        ///////////////////             TEST
        // double *d_ADn = NULL;
        // checkCudaErrors(cudaMalloc((void **)&d_ADn, sizeof(double) * ADA_nrows * ADA_ncols));

        // cusparseMatDescr_t matGenDescr;
        // checkCudaErrors(cusparseCreateMatDescr(&matGenDescr));
        // checkCudaErrors(cusparseSetMatType(matGenDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
        // checkCudaErrors(cusparseSetMatIndexBase(matGenDescr, CUSPARSE_INDEX_BASE_ZERO));

        // checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, ADA_nrows, ADA_ncols,
        //                                    matGenDescr, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
        //                                    d_ADAvals, d_ADAoffs, d_ADAinds,
        //                                    d_ADn, ADA_nrows));

        // printf("%d) MAT\n", iterations);
        // utils_printDmat(ADA_nrows, ADA_ncols, ADA_nrows, d_ADn, true, false);
        // checkCudaErrors(cudaFree(d_ADn));

        // printf("OFFS:\n");
        // utils_printIvec(A_nrows+1, d_csrAOffs, true);
        // printf("INDS:\n");
        // utils_printIvec(A_nnz, d_csrAInds, true);
        // printf("VALS:\n");
        // utils_printDvec(A_nnz, d_csrAVals, true);
        ///////////////////             END TEST

        ///////////////             COMPUTE TMPA = - AD * resC - resB

        // store TMPA vector on buffer 1, size of buffer 1 is guaranteed to 
        // be >= 2*ncols, copy resB on TMPA
        d_tmpA = d_buffer1;
        d_tmpB = &d_buffer1[node.ncols];
        checkCudaErrors(cusparseCreateDnVec(&vecTmpA, AD_nrows, d_tmpA, CUDA_R_64F));
        checkCudaErrors(cudaMemcpyAsync(d_tmpA, d_resB, sizeof(double) * AD_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        // compute TMPB
        elem_mult_kernel<<<bSize, 32>>>(d_resXS, d_invS, d_tmpB, node.ncols);
        checkCudaErrors(cusparseCreateDnVec(&vecTmpB, AD_ncols, d_tmpB, CUDA_R_64F));

        {    
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AD_descr, vecResC,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AD_descr, vecResC,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));
        }

        ///////////////             COMPUTE TMPA = TMPA + A * TMPB
        {    
            alpha = 1.0;
            beta = 1.0;

            cudaDeviceSynchronize();

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, node.matDescr, vecTmpB,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, node.matDescr, vecTmpB,
                        &beta, vecTmpA, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));
        }

        ///////////////             COMPUTE DEL_Y = SOLVE(ADA, TMPA)
        {
            checkCudaErrors(cusolverSpDcsrlsvchol(node.cusolverSpHandle,
                                            ADA_nrows, ADA_nnz, matDescrGen,
                                            d_ADAvals, d_ADAoffs, d_ADAinds,
                                            d_tmpA,
                                            node.env->MEHROTRA_CHOL_TOL, reorder,
                                            d_delY, &singularity));
        }

        ///////////////             COMPUTE DEL_S = - AT * DEL_Y - resC
        checkCudaErrors(cudaMemcpyAsync(d_delS, d_resC, sizeof(double) * AT_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        {
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AT_descr, vecDelY,
                        &beta, vecDelS, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, AT_descr, vecDelY,
                        &beta, vecDelS, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));   
        }

        ///////////////             COMPUTE DEL_X = -TMP_B - D * DEL_S
        checkCudaErrors(cudaMemcpyAsync(d_delX, d_tmpB, sizeof(double) * D_nrows, cudaMemcpyDeviceToDevice, node.cudaStream));

        {
            alpha = -1.0;
            beta = -1.0;

            // buffer 1 used for TMPA and TMPB
            checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, D_descr, vecDelS,
                        &beta, vecDelX, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        &bufferSize2));

            if (bufferSize2 > currBufferSize2)
            {
                currBufferSize2 = bufferSize2;
                if (d_buffer2) checkCudaErrors(cudaFree(d_buffer2));
                checkCudaErrors(cudaMalloc((void **)&d_buffer2, currBufferSize2));
            }

            checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, D_descr, vecDelS,
                        &beta, vecDelX, CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                        d_buffer2));   
        }

        // printf("%d) AFFINE SYSTEM\n", iterations);
        // printf("delta X\n");
        // utils_printDvec(node.ncols, d_delX, true);
        // printf("delta Y\n");
        // utils_printDvec(node.nrows, d_delY, true);
        // printf("delta S\n");
        // utils_printDvec(node.ncols, d_delS, true);

        // affine step length, definition 14.32 at page 408(427)
        // alpha_max_p = min([-xi / delta_xi for xi, delta_xi in zip(x, delta_x_aff) if delta_xi < 0.0])
        // alpha_max_d = min([-si / delta_si for si, delta_si in zip(s, delta_s_aff) if delta_si < 0.0])

        // finding alphaMaxPrim and alphaMaxDual: to improve
        find_alpha_max(&alphaMaxPrim, &alphaMaxDual,
                       d_x, d_delX, d_s, d_delS, node.ncols);

        alphaPrim = gsl_min(1.0, node.env->MEHROTRA_ETA * alphaMaxPrim);
        alphaDual = gsl_min(1.0, node.env->MEHROTRA_ETA * alphaMaxDual);

        // d_deltaX, d_deltaY, d_deltaS are pointees to d_sol
        // the solution of the previous system 

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaPrim, d_delX, 1, d_x, 1));

        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.nrows,
                                    &alphaDual, d_delY, 1, d_y, 1));
        
        checkCudaErrors(cublasDaxpy(node.cublasHandle, node.ncols,
                                    &alphaDual, d_delS, 1, d_s, 1));

        ///////////////             UPDATE

        alpha = -(alphaDual - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.ncols,
                                    &alpha, d_resC, 1));

        alpha = -(alphaPrim - 1.0);
        checkCudaErrors(cublasDscal(node.cublasHandle, node.nrows,
                                    &alpha, d_resB, 1));

        checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols, d_x, 1, d_s, 1, &mu));
        mu /= node.ncols;

        ++iterations;
    }

    node.iterations = iterations;
    
    checkCudaErrors(cublasDdot(node.cublasHandle, node.ncols,
                               d_x, 1, node.d_ObjDns, 1, &node.objvalPrim));

    checkCudaErrors(cublasDdot(node.cublasHandle, node.nrows,
                               d_y, 1, node.d_RhsDns, 1, &node.objvalDual));

    node.env->logger("Mehrotra procedure complete", "INFO", 10);
    node.timeSolverEnd = node.env->timer();

    ///////////////////             RELEASE MEMORY

    free(node.h_x);
    free(node.h_y);
    free(node.h_s);

    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_s));

    checkCudaErrors(cudaFree(d_delX));
    checkCudaErrors(cudaFree(d_delY));
    checkCudaErrors(cudaFree(d_delS));
    
    checkCudaErrors(cudaFree(d_resC));
    checkCudaErrors(cudaFree(d_resB));
    checkCudaErrors(cudaFree(d_resXS));
    
    checkCudaErrors(cudaFree(d_invS));
    
    checkCudaErrors(cusparseDestroyMatDescr(matDescrGen));

    cusparseDestroySpMat(AT_descr);
    cusparseDestroySpMat(D_descr);
    cusparseDestroySpMat(AD_descr);
    cusparseDestroySpMat(ADA_descr);

    checkCudaErrors(cudaFree(d_AToffs));
    checkCudaErrors(cudaFree(d_ATinds));
    checkCudaErrors(cudaFree(d_ATvals));
    
    checkCudaErrors(cudaFree(d_Dvals));
    checkCudaErrors(cudaFree(d_Doffs));
    checkCudaErrors(cudaFree(d_Dinds));

    checkCudaErrors(cudaFree(d_ADvals));
    checkCudaErrors(cudaFree(d_ADoffs));
    checkCudaErrors(cudaFree(d_ADinds));

    checkCudaErrors(cudaFree(d_ADAvals));
    checkCudaErrors(cudaFree(d_ADAoffs));
    checkCudaErrors(cudaFree(d_ADAinds));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_init_1(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;

    int64_t AAT_nrows = node.nrows, AAT_ncols = node.nrows, AAT_nnz = 0;
    double alpha = 1.0;
    double beta = 0.0;

    int *AAT_inds = NULL, *AAT_offs = NULL;
    double *AAT_vals = NULL;

    void *d_buffer1 = NULL, *d_buffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    cusparseSpMatDescr_t AAT_descr;
    cusparseMatDescr_t AAT_descrGen, matTransDescrGen;
    cusparseSpGEMMDescr_t spgemmDescr;

    checkCudaErrors(cusparseCreateMatDescr(&AAT_descrGen));
    checkCudaErrors(cusparseSetMatType(AAT_descrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(AAT_descrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateMatDescr(&matTransDescrGen));
    checkCudaErrors(cusparseSetMatType(matTransDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matTransDescrGen, CUSPARSE_INDEX_BASE_ZERO));

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

    // copy the final products to the matrix AAT
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

    /*void *d_b, *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_b, AAT_nrows*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_x, AAT_nrows*sizeof(double)));

    checkCudaErrors(cusolverSpDcsrlsvchol(
        node.cusolverSpHandle, AAT_nrows, AAT_nnz,
        AAT_descrGen, AAT_vals, AAT_offs, AAT_inds,
        (double*)d_b, node.env->MEHROTRA_CHOL_TOL, reorder, (double*)d_x, &singularity));

    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_x));*/

    cusparseSpMatGetSize(AAT_descr, &r, &c, &n);
    std::cout << "\nAAT" << std::endl;
    std::cout << "rows: " << r << ", cols: " << c << ", num nz: " << n << std::endl;

    ///////////////////             COMPUTE s = - mat' * y + obj
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.d_s, node.d_ObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseCsrmvEx_bufferSize(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               node.ncols, node.nrows, node.nnz,
                                               &alpha, CUDA_R_64F,
                                               matTransDescrGen,
                                               node.d_csrMatTransVals, CUDA_R_64F,
                                               node.d_csrMatTransOffs,
                                               node.d_csrMatTransInds,
                                               node.d_y, CUDA_R_64F,
                                               &beta, CUDA_R_64F,
                                               node.d_s, CUDA_R_64F, CUDA_R_64F,
                                               &bufferSize1));

    checkCudaErrors(cudaMalloc((void **)&d_buffer1, bufferSize1));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseCsrmvEx(node.cusparseHandle, CUSPARSE_ALG_MERGE_PATH,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    node.ncols, node.nrows, node.nnz,
                                    &alpha, CUDA_R_64F,
                                    matTransDescrGen,
                                    node.d_csrMatTransVals, CUDA_R_64F,
                                    node.d_csrMatTransOffs,
                                    node.d_csrMatTransInds,
                                    node.d_y, CUDA_R_64F,
                                    &beta, CUDA_R_64F,
                                    node.d_s, CUDA_R_64F, CUDA_R_64F,
                                    d_buffer1));

    checkCudaErrors(cusparseDestroyMatDescr(AAT_descrGen));
    checkCudaErrors(cusparseDestroyMatDescr(matTransDescrGen));
    checkCudaErrors(cusparseDestroySpMat(AAT_descr));

    checkCudaErrors(cudaFree(d_buffer1));
    checkCudaErrors(cudaFree(d_buffer2));

    checkCudaErrors(cudaFree(AAT_inds));
    checkCudaErrors(cudaFree(AAT_offs));
    checkCudaErrors(cudaFree(AAT_vals));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_init_2(SyphaNodeSparse &node)
{
    const int reorder = 0;
    int singularity = 0;
    int info = 0;
    int i = 0;
    int I_matBytes = node.nrows * node.nrows * sizeof(double);

    double alpha = 1.0;
    double beta = 0.0;

    int *d_ipiv = NULL;
    double *d_AAT = NULL;
    double *d_matDn = NULL;
    double *h_I = NULL;

    void *d_buffer = NULL;
    size_t currBufferSize = 0;
    size_t bufferSize = 0;
    char message[1024];

    cusolverDnParams_t cusolverDnParams;
    cusparseDnVecDescr_t vecX, vecY, vecS;
    cusparseDnMatDescr_t AAT_descr, matDnDescr;
    cusparseMatDescr_t matDescrGen;

    node.env->logger("Mehrotra starting point computation", "INFO", 13);
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

    checkCudaErrors(cusparseCreateMatDescr(&matDescrGen));
    checkCudaErrors(cusparseSetMatType(matDescrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(matDescrGen, CUSPARSE_INDEX_BASE_ZERO));

    checkCudaErrors(cusparseCreateDnVec(&vecX, (int64_t)node.ncols,
                                        node.d_x, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecY, (int64_t)node.nrows,
                                        node.d_y, CUDA_R_64F));

    checkCudaErrors(cusparseCreateDnVec(&vecS, (int64_t)node.ncols,
                                        node.d_s, CUDA_R_64F));

    checkCudaErrors(cudaMalloc((void **)&d_AAT, sizeof(double) * node.nrows * node.nrows));
    checkCudaErrors(cudaMalloc((void **)&d_matDn, sizeof(double) * node.nrows * node.ncols));

    checkCudaErrors(cusparseCreateDnMat(&AAT_descr, (int64_t)node.nrows, (int64_t)node.nrows,
                                        (int64_t)node.nrows, d_AAT, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    checkCudaErrors(cusparseCreateDnMat(&matDnDescr, (int64_t)node.nrows, (int64_t)node.ncols,
                                        (int64_t)node.nrows, d_matDn, CUDA_R_64F,
                                        CUSPARSE_ORDER_COL));

    ///////////////////             STORE MATRIX IN DENSE FORMAT
    node.env->logger("solver_sparse_mehrotra_init - storing matrix in dense format", "INFO", 20);
    checkCudaErrors(cusparseDcsr2dense(node.cusparseHandle, node.nrows, node.ncols,
                                       matDescrGen, // CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
                                       node.d_csrMatVals, node.d_csrMatOffs, node.d_csrMatInds,
                                       d_matDn, node.nrows));

    ///////////////////             COMPUTE AAT INVERSE MATRIX

    // GEMM Computation: MATRIX * MATRIX'
    node.env->logger("solver_sparse_mehrotra_init - computing mat * mat'", "INFO", 20);
    checkCudaErrors(cusparseSpMM_bufferSize(node.cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, matDnDescr,
                                            &beta, AAT_descr,
                                            CUDA_R_64F,
                                            CUSPARSE_CSRMM_ALG1,
                                            &bufferSize));

    // allocate memory for computation
    currBufferSize = bufferSize > I_matBytes ? bufferSize : I_matBytes;
    checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));

    checkCudaErrors(cusparseSpMM(node.cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, matDnDescr,
                                 &beta, AAT_descr,
                                 CUDA_R_64F,
                                 CUSPARSE_CSRMM_ALG1,
                                 d_buffer));

    ///////////////////             MATRIX INVERSION

    node.env->logger("solver_sparse_mehrotra_init - computing matrix inversion", "INFO", 20);
    // See https://stackoverflow.com/questions/50892906/what-is-the-most-efficient-way-to-compute-the-inverse-of-a-general-matrix-using
    checkCudaErrors(cusolverDnDgetrf_bufferSize(node.cusolverDnHandle,
                                                node.nrows, node.nrows,
                                                d_AAT, node.nrows,
                                                (int *)&bufferSize));

    // allocate memory for computation
    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }
    checkCudaErrors(cudaMalloc((void **)&d_ipiv, sizeof(int) * node.nrows));

    /*checkCudaErrors(cusolverDnDgetrf(node.cusolverDnHandle,
                                     node.nrows, node.nrows,
                                     d_AAT, node.nrows,
                                     (double *)d_buffer, d_ipiv,
                                     &info));*/

    printf("AAT after getrf\n");
    utils_printDmat(node.nrows, node.nrows, node.nrows, d_AAT, true, true);

    sprintf(message, "solver_sparse_mehrotra_init - cusolverDnGetrf returned %d", info);
    node.env->logger(message, "INFO", 20);

    // set I matrix
    h_I = (double *)calloc(node.nrows * node.nrows, sizeof(double));
    for (i = 0; i < node.nrows; ++i)
    {
        h_I[node.nrows * i + i] = 1.0;
    }
    //checkCudaErrors(cudaMemcpyAsync(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice, node.cudaStream));
    //checkCudaErrors(cudaMemcpy(d_buffer, h_I, sizeof(double) * node.nrows * node.nrows, cudaMemcpyHostToDevice));
    free(h_I);

    checkCudaErrors(cusolverDnDgetrs(node.cusolverDnHandle, CUBLAS_OP_N,
                                     node.nrows, node.nrows,
                                     d_AAT, node.nrows,
                                     d_ipiv,
                                     (double *)d_buffer, node.nrows,
                                     &info));

    printf("AAT after getrs\n");
    utils_printDmat(node.nrows, node.nrows, node.nrows, d_AAT, true, true);

    sprintf(message, "solver_sparse_mehrotra_init - cusolverDnGetrs returned %d", info);
    node.env->logger(message, "INFO", 20);

    /*void *d_b, *d_x;
    checkCudaErrors(cudaMalloc((void **)&d_b, AAT_nrows*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_x, AAT_nrows*sizeof(double)));

    checkCudaErrors(cusolverSpDcsrlsvchol(
        node.cusolverSpHandle, AAT_nrows, AAT_nnz,
        matDescrGen, AAT_vals, AAT_offs, AAT_inds,
        (double*)d_b, node.env->MEHROTRA_CHOL_TOL, reorder, (double*)d_x, &singularity));

    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_x));*/

    ///////////////////             COMPUTE s = - mat' * y + obj
    node.env->logger("solver_sparse_mehrotra_init - computing s = - mat' * y + obj", "INFO", 20);
    alpha = -1.0;
    beta = 1.0;

    // copy obj on s
    checkCudaErrors(cudaMemcpyAsync(node.d_s, node.d_ObjDns, sizeof(double) * node.ncols,
                                    cudaMemcpyDeviceToDevice, node.cudaStream));

    checkCudaErrors(cusparseSpMV_bufferSize(node.cusparseHandle,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &alpha, node.matDescr, vecY,
                                            &beta, vecS,
                                            CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                            &bufferSize));

    if (bufferSize > currBufferSize)
    {
        currBufferSize = bufferSize;
        checkCudaErrors(cudaMalloc((void **)&d_buffer, currBufferSize));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cusparseSpMV(node.cusparseHandle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, node.matDescr, vecY,
                                 &beta, vecS,
                                 CUDA_R_64F, CUSPARSE_CSRMV_ALG2,
                                 d_buffer));

    ///////////////////             FREE RESOURCES
    checkCudaErrors(cusolverDnDestroyParams(cusolverDnParams));

    checkCudaErrors(cusparseDestroyMatDescr(matDescrGen));
    checkCudaErrors(cusparseDestroyDnMat(AAT_descr));

    checkCudaErrors(cusparseDestroyDnVec(vecX));
    checkCudaErrors(cusparseDestroyDnVec(vecY));
    checkCudaErrors(cusparseDestroyDnVec(vecS));

    checkCudaErrors(cudaFree(d_ipiv));
    checkCudaErrors(cudaFree(d_buffer));

    checkCudaErrors(cudaFree(d_AAT));
    checkCudaErrors(cudaFree(d_matDn));

    return CODE_SUCCESFULL;
}

SyphaStatus solver_sparse_mehrotra_init_gsl(SyphaNodeSparse &node)
{
    int i, j;
    int signum = 0;
    double deltaX, deltaS, prod, sumX, sumS;
    char message[1024];

    gsl_vector *x = NULL;
    gsl_vector *y = NULL;
    gsl_vector *s = NULL;
    gsl_matrix *inv = NULL;
    gsl_matrix *mat = NULL;
    gsl_matrix *tmp = NULL;
    gsl_permutation *perm = NULL;

    x = gsl_vector_alloc((size_t)node.ncols);
    y = gsl_vector_alloc((size_t)node.nrows);
    s = gsl_vector_alloc((size_t)node.ncols);
    inv = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.nrows);
    mat = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.ncols);
    tmp = gsl_matrix_calloc((size_t)node.nrows, (size_t)node.ncols);
    perm = gsl_permutation_alloc((size_t)node.nrows);

    // csr to dense
    for (i = 0; i < node.nrows; ++i)
    {
        for (j = node.h_csrMatOffs->data()[i]; j < node.h_csrMatOffs->data()[i + 1]; ++j)
        {
            mat->data[node.ncols * i + node.h_csrMatInds->data()[j]] = node.h_csrMatVals->data()[j];
        }
    }
    //printf("MAT:\n");
    //utils_printDmat(node.nrows, node.ncols, node.ncols, mat->data, false);

    ///////////////////             MATRIX MULT
    node.env->logger("solver_sparse_mehrotra_init - computing A * A'", "INFO", 20);
    mat->size1 = node.nrows;
    mat->size2 = node.ncols;
    mat->tda = node.ncols;
    tmp->size1 = node.nrows;
    tmp->size2 = node.nrows;
    tmp->tda = node.ncols;
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, mat, mat, 0.0, tmp);

    //printf("AAT:\n");
    //utils_printDmat(node.nrows, node.nrows, node.ncols, tmp->data, false);

    ///////////////////             MATRIX INVERSION
    node.env->logger("solver_sparse_mehrotra_init - computing inv(AAT)", "INFO", 20);
    
    inv->size1 = node.nrows;
    inv->size2 = node.nrows;
    inv->tda = node.nrows;
    gsl_linalg_LU_decomp(tmp, perm, &signum);
    gsl_linalg_LU_invert(tmp, perm, inv);

    //printf("INV:\n");
    //utils_printDmat(node.nrows, node.nrows, node.nrows, inv->data, false);

    ///////////////////             COMPUTE x = mat' * AAT_inv * rhs
    node.env->logger("solver_sparse_mehrotra_init - computing x <-- A' * inv(AAT) * rhs", "INFO", 20);

    tmp->size1 = node.ncols;
    tmp->size2 = node.nrows;
    tmp->tda = node.nrows;
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, mat, inv, 0.0, tmp);

    // put RHS in Y
    memcpy(y->data, node.h_RhsDns, sizeof(double) * node.nrows);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, y, 0.0, x);

    //printf("TMP:\n");
    //utils_printDmat(node.ncols, node.nrows, node.nrows, tmp->data, false);

    ///////////////////             COMPUTE y = AAT_inv * mat * obj
    node.env->logger("solver_sparse_mehrotra_init - computing y <-- inv(AAT) * A * obj", "INFO", 20);

    tmp->size1 = node.nrows;
    tmp->size2 = node.ncols;
    tmp->tda = node.ncols;
    
    // put OBJ in S
    memcpy(s->data, node.h_ObjDns, sizeof(double) * node.ncols);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, mat, 0.0, tmp);
    gsl_blas_dgemv(CblasNoTrans, 1.0, tmp, s, 0.0, y);

    //printf("TMP:\n");
    //utils_printDmat(node.ncols, node.nrows, node.nrows, tmp->data, false);

    ///////////////////             COMPUTE s = - mat' * y + obj
    node.env->logger("solver_sparse_mehrotra_init - computing s <-- obj - A' * y", "INFO", 20);
    gsl_blas_dgemv(CblasTrans, -1.0, mat, y, 1.0, s);

    deltaX = gsl_max(-1.5 * gsl_vector_min(x), 0.0);
    deltaS = gsl_max(-1.5 * gsl_vector_min(s), 0.0);

    gsl_vector_add_constant(x, deltaX);
    gsl_vector_add_constant(s, deltaS);

    gsl_blas_ddot(x, s, &prod);
    prod *= 0.5;

    sumX = 0.0;
    sumS = 0.0;
    for (j = 0; j < node.ncols; ++j)
    {
        sumX += x->data[j];
        sumS += s->data[j];
    }
    deltaX = prod / sumS;
    deltaS = prod / sumX;

    gsl_vector_add_constant(x, deltaX);
    gsl_vector_add_constant(s, deltaS);

    //printf("X:\n");
    //utils_printDvec(node.ncols, x->data, false);
    //printf("Y:\n");
    //utils_printDvec(node.nrows, y->data, false);
    //printf("S:\n");
    //utils_printDvec(node.ncols, s->data, false);

    memcpy(node.h_x, x->data, sizeof(double) * node.ncols);
    memcpy(node.h_y, y->data, sizeof(double) * node.nrows);
    memcpy(node.h_s, s->data, sizeof(double) * node.ncols);

    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(s);
    gsl_matrix_free(inv);
    gsl_matrix_free(mat);
    gsl_matrix_free(tmp);
    gsl_permutation_free(perm);

    return CODE_SUCCESFULL;
}
