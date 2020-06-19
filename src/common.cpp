
#include "common.h"

int utils_printDmat(int m, int n, int l, double *mat, bool device)
{
    double *matLoc;
    if (device)
    {
        matLoc = (double *)malloc(sizeof(double) * l * n);
        checkCudaErrors(cudaMemcpy(matLoc, mat, sizeof(double) * l * n, cudaMemcpyDeviceToHost));
    } else {
        matLoc = mat;
    }
    
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < l; ++j)
        {
            printf("%6.2lf ", matLoc[l * i + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    if (device)
    {
        free(matLoc);
    }
    
    return 0;
}

int utils_printDvec(int n, double *vec, bool device)
{
    double *vecLoc;
    if (device)
    {
        vecLoc = (double *)malloc(sizeof(double) * n);
        checkCudaErrors(cudaMemcpy(vecLoc, vec, sizeof(double) * n, cudaMemcpyDeviceToHost));
    }
    else
    {
        vecLoc = vec;
    }

    for (int i = 0; i < n; ++i)
    {
            printf("%6.2lf ", vecLoc[i]);
    }
    printf("\n");

    if (device)
    {
        free(vecLoc);
    }

    return 0;
}
