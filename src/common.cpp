
#include "common.h"

#define ZERO_TOL 1.E-20

int utils_printDmat(int m, int n, int l, double *mat, bool device, bool trans)
{
    double val;
    double *matLoc;
    if (device)
    {
        matLoc = (double *)malloc(sizeof(double) * l * n);
        checkCudaErrors(cudaMemcpy(matLoc, mat, sizeof(double) * l * n, cudaMemcpyDeviceToHost));
    }
    else
    {
        matLoc = mat;
    }

    m = trans ? l : m;
    l = trans ? m : l;
    printf("[");
    for (int i = 0; i < m; ++i)
    {
        printf("[");
        for (int j = 0; j < l; ++j)
        {
            val = trans ? matLoc[l * j + i] : matLoc[l * i + j];
            val = abs(val) < ZERO_TOL ? 0.0 : val;
            printf("%8.6lf, ", val);
        }
        printf("],\n");
    }
    printf("]\n");

    if (device)
    {
        free(matLoc);
    }

    return 0;
}

int utils_printIvec(int n, int *vec, bool device)
{
    int *vecLoc;
    if (device)
    {
        vecLoc = (int *)malloc(sizeof(int) * n);
        checkCudaErrors(cudaMemcpy(vecLoc, vec, sizeof(int) * n, cudaMemcpyDeviceToHost));
    }
    else
    {
        vecLoc = vec;
    }

    printf("[");
    for (int i = 0; i < n; ++i)
    {
        printf("%4d, ", vecLoc[i]);
    }
    printf("]\n");

    if (device)
    {
        free(vecLoc);
    }

    return 0;
}

int utils_printDvec(int n, double *vec, bool device)
{
    double val;
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

    printf("[");
    for (int i = 0; i < n; ++i)
    {
        val = abs(vecLoc[i]) < ZERO_TOL ? 0.0 : vecLoc[i];
        printf("%8.6lf, ", val);
    }
    printf("]\n");

    if (device)
    {
        free(vecLoc);
    }

    return 0;
}
