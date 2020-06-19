
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
    
    for (int i = 0; i < l; ++i)
    {
        for (int j = 0; j < n; ++j)
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
