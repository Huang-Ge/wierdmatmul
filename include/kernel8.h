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

#ifndef _KNL8_H_
#define _KNL8_H_

#include "utils.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 512
#define K_BLOCKING 384

void scale_c_k8(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_opt_k8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k8(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            double tmp=C(i,j);
            for (k=0;k<K;k++){
                tmp += alpha*wierdMul(A(i,k), B(k,j));
            }
            C(i,j) = tmp;
        }
    }
}

void macro_dummy_cpu_v8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC){
    int i,j,k;
    // if (beta != 1.0) scale_c_k8(C,M,N,LDC,beta);
    int M4=M&-4,N4=N&-4;
    for (i=0;i<M4;i+=4){
        for (j=0;j<N4;j+=4){
            double c00=C(i,j);
            double c01=C(i,j+1);
            double c02=C(i,j+2);
            double c03=C(i,j+3);
            double c10=C(i+1,j);
            double c11=C(i+1,j+1);
            double c12=C(i+1,j+2);
            double c13=C(i+1,j+3);
            double c20=C(i+2,j);
            double c21=C(i+2,j+1);
            double c22=C(i+2,j+2);
            double c23=C(i+2,j+3);
            double c30=C(i+3,j);
            double c31=C(i+3,j+1);
            double c32=C(i+3,j+2);
            double c33=C(i+3,j+3);
            for (k=0;k<K;k++){
                double a0 = alpha*A(i,k);
                double a1 = alpha*A(i+1,k);
                double a2 = alpha*A(i+2,k);
                double a3 = alpha*A(i+3,k);
                double b0 = B(k,j);
                double b1 = B(k,j+1);
                double b2 = B(k,j+2);
                double b3 = B(k,j+3);
                c00 += wierdMul(a0, b0);
                c01 += wierdMul(a0, b1);
                c02 += wierdMul(a0, b2);
                c03 += wierdMul(a0, b3);
                c10 += wierdMul(a1, b0);
                c11 += wierdMul(a1, b1);
                c12 += wierdMul(a1, b2);
                c13 += wierdMul(a1, b3);
                c20 += wierdMul(a2, b0);
                c21 += wierdMul(a2, b1);
                c22 += wierdMul(a2, b2);
                c23 += wierdMul(a2, b3);
                c30 += wierdMul(a3, b0);
                c31 += wierdMul(a3, b1);
                c32 += wierdMul(a3, b2);
                c33 += wierdMul(a3, b3);
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i,j+2) = c02;
            C(i,j+3) = c03;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
            C(i+1,j+2) = c12;
            C(i+1,j+3) = c13;
            C(i+2,j) = c20;
            C(i+2,j+1) = c21;
            C(i+2,j+2) = c22;
            C(i+2,j+3) = c23;
            C(i+3,j) = c30;
            C(i+3,j+1) = c31;
            C(i+3,j+2) = c32;
            C(i+3,j+3) = c33;
        }
    }
    if (M4==M&&N4==N) return;
    // boundary conditions
    if (M4!=M) mydgemm_cpu_opt_k8(M-M4,N,K,alpha,A+M4,LDA,B,LDB,1.0,&C(M4,0),LDC);
    if (N4!=N) mydgemm_cpu_opt_k8(M4,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}

void mydgemm_cpu_v8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    
    if (beta != 1.0) scale_c_k8(C,M,N,LDC,beta);
    
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                //macro kernel: to compute C += A_tilt * B_tilt
                macro_dummy_cpu_v8(m_inc,n_inc,k_inc,alpha,&A(m_count,k_count), LDA, &B(k_count,n_count), LDB, &C(m_count, n_count), LDC);
            }
        }
    }

}

#endif