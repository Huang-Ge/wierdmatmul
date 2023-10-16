#include <stdio.h>
#include <stdbool.h>
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kernels.h"
void print_vector(double *vec, int n){
    int i;
    for (i = 0; i < n; i++){
        printf(" %5.2f, ", vec[i]);
    }
    printf("\n");
}

void print_matrix_lda(const double *A, int lda, int m, int n)
{
    int i,j;
    const double *a_curr_pos=A;
    printf("[");
    for (i = 0; i < n; i++){
        for (j=0;j<m;j++){
            printf("%5.2f, ",*a_curr_pos);
            a_curr_pos++;
        }
        a_curr_pos+=(lda-m);
    }
    printf("]\n");
}

void print_matrix(const double *A, int m, int n){
    int i;printf("[");
    for (i = 0; i < m * n; i++){
        if ((i + 1) % n == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % n == 0){
            if (i + 1 < m * n)
                printf(";\n");
        }
    }
    printf("]\n");
}

double get_sec(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + 1e-6 * time.tv_usec);
}

void randomize_matrix(double *A, int m, int n){
    srand(time(NULL));
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            A[i * n + j] = (double)(rand() % 100) + 0.01 * (rand() % 100);
            if (rand() % 2 == 0) A[i * n + j] *= 1.0;
        }
    }
}

void copy_matrix(double *src, double *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++){
        *(dest + i) = *(src + i);
    }
    if (i != n){
        printf("copy failed at %d while there are %d elements in total.\n", i, n);
    }
}

bool verify_matrix(double *mat1, double *mat2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++){
        diff = fabs(mat1[i] - mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}


void test_dummy_v1(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    dummy_cpu_v1(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_dummy_v2(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    dummy_cpu_v2(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_kernel(int kernel_num,int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    switch (kernel_num){
        case 1: test_dummy_v1(m,n,k,alpha,A,B,beta,C); break;
        case 2: test_dummy_v2(m,n,k,alpha,A,B,beta,C); break;
        default: break;
    }
}