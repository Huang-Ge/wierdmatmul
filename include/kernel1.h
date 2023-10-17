/*
 * Copyright (C) [year the code was published or updated] by [original author's name or GitHub username].
 *
 * This code is sourced from the repository at:
 * https://github.com/yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F
 *
 * All rights to the original code belong to the author. The use, distribution,
 * or reproduction of this code is subject to the terms specified by the original
 * author and/or the associated license.
 */

#ifndef _KNL1_H_
#define _KNL1_H_

#include "utils.h"

#define A(i,j) A[(i)*LDA+(j)]
#define B(i,j) B[(i)*LDB+(j)]
#define C(i,j) C[(i)*LDC+(j)]

void scale_c_k1(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void dummy_cpu_v1(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k1(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            for (k=0;k<K;k++){
                C(i,j) += alpha*wierdMul(A(i,k), B(k,j));
            }
        }
    }
}

#endif // _KNL1_H_