#include "sypha_solver.h"
#include "sypha_solver_krylov.h"
#include "sypha_cuda_helper.h"

void initializeIpmWorkspace(IpmWorkspace *ws, int maxKktNrows, int maxKktNnz, int maxNcols)
{
    if (maxKktNnz > ws->kktNnzCapacity)
    {
        if (ws->d_csrAInds) checkCudaErrors(cudaFree(ws->d_csrAInds));
        if (ws->d_csrAVals) checkCudaErrors(cudaFree(ws->d_csrAVals));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_csrAInds), sizeof(int) * static_cast<size_t>(maxKktNnz)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_csrAVals), sizeof(double) * static_cast<size_t>(maxKktNnz)));
        ws->kktNnzCapacity = maxKktNnz;
    }
    if (maxKktNrows > ws->kktNrowsCapacity)
    {
        if (ws->d_csrAOffs) checkCudaErrors(cudaFree(ws->d_csrAOffs));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_csrAOffs), sizeof(int) * static_cast<size_t>(maxKktNrows + 1)));
        ws->kktNrowsCapacity = maxKktNrows;
    }
    if (maxKktNrows > ws->vectorCapacity)
    {
        if (ws->d_rhs) checkCudaErrors(cudaFree(ws->d_rhs));
        if (ws->d_sol) checkCudaErrors(cudaFree(ws->d_sol));
        if (ws->d_prevSol) checkCudaErrors(cudaFree(ws->d_prevSol));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_rhs), sizeof(double) * static_cast<size_t>(maxKktNrows)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_sol), sizeof(double) * static_cast<size_t>(maxKktNrows)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_prevSol), sizeof(double) * static_cast<size_t>(maxKktNrows)));
        ws->vectorCapacity = maxKktNrows;
    }
    const int nBlocks = (maxNcols + 255) / 256;
    if (maxNcols > ws->alphaCapacity)
    {
        if (ws->d_tmp_prim) checkCudaErrors(cudaFree(ws->d_tmp_prim));
        if (ws->d_tmp_dual) checkCudaErrors(cudaFree(ws->d_tmp_dual));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_tmp_prim), sizeof(double) * static_cast<size_t>(maxNcols)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_tmp_dual), sizeof(double) * static_cast<size_t>(maxNcols)));
        ws->alphaCapacity = maxNcols;
    }
    if (nBlocks > ws->alphaBlocksCapacity)
    {
        if (ws->d_blockmin_prim) checkCudaErrors(cudaFree(ws->d_blockmin_prim));
        if (ws->d_blockmin_dual) checkCudaErrors(cudaFree(ws->d_blockmin_dual));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_blockmin_prim), sizeof(double) * static_cast<size_t>(nBlocks)));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_blockmin_dual), sizeof(double) * static_cast<size_t>(nBlocks)));
        ws->alphaBlocksCapacity = nBlocks;
    }
    if (!ws->d_alphaResult)
    {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ws->d_alphaResult), sizeof(double) * 2));
    }
    if (!ws->A_descr)
    {
        checkCudaErrors(cusparseCreateMatDescr(&ws->A_descr));
        checkCudaErrors(cusparseSetMatType(ws->A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(ws->A_descr, CUSPARSE_INDEX_BASE_ZERO));
    }
    ws->isAllocated = true;
}

void releaseIpmWorkspace(IpmWorkspace *ws)
{
    if (ws->d_csrAInds) { checkCudaErrors(cudaFree(ws->d_csrAInds)); ws->d_csrAInds = nullptr; }
    if (ws->d_csrAOffs) { checkCudaErrors(cudaFree(ws->d_csrAOffs)); ws->d_csrAOffs = nullptr; }
    if (ws->d_csrAVals) { checkCudaErrors(cudaFree(ws->d_csrAVals)); ws->d_csrAVals = nullptr; }
    if (ws->d_rhs) { checkCudaErrors(cudaFree(ws->d_rhs)); ws->d_rhs = nullptr; }
    if (ws->d_sol) { checkCudaErrors(cudaFree(ws->d_sol)); ws->d_sol = nullptr; }
    if (ws->d_prevSol) { checkCudaErrors(cudaFree(ws->d_prevSol)); ws->d_prevSol = nullptr; }
    if (ws->d_tmp_prim) { checkCudaErrors(cudaFree(ws->d_tmp_prim)); ws->d_tmp_prim = nullptr; }
    if (ws->d_tmp_dual) { checkCudaErrors(cudaFree(ws->d_tmp_dual)); ws->d_tmp_dual = nullptr; }
    if (ws->d_blockmin_prim) { checkCudaErrors(cudaFree(ws->d_blockmin_prim)); ws->d_blockmin_prim = nullptr; }
    if (ws->d_blockmin_dual) { checkCudaErrors(cudaFree(ws->d_blockmin_dual)); ws->d_blockmin_dual = nullptr; }
    if (ws->d_alphaResult) { checkCudaErrors(cudaFree(ws->d_alphaResult)); ws->d_alphaResult = nullptr; }
    if (ws->d_buffer) { checkCudaErrors(cudaFree(ws->d_buffer)); ws->d_buffer = nullptr; }
    if (ws->A_descr) { checkCudaErrors(cusparseDestroyMatDescr(ws->A_descr)); ws->A_descr = nullptr; }
    if (ws->krylov)
    {
        releaseKrylovWorkspace(ws->krylov);
        delete ws->krylov;
        ws->krylov = nullptr;
    }
    ws->isAllocated = false;
    ws->kktNnzCapacity = 0;
    ws->kktNrowsCapacity = 0;
    ws->vectorCapacity = 0;
    ws->alphaCapacity = 0;
    ws->alphaBlocksCapacity = 0;
    ws->bufferCapacity = 0;
}
