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

#ifndef _KNL9_H_
#define _KNL9_H_

#include "utils.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 512
#define K_BLOCKING 384

#define KERNEL_K9_8x8_naive\
    a0 = alpha*A(i,k);\
    a1 = alpha*A(i+1,k);\
    a2 = alpha*A(i+2,k);\
    a3 = alpha*A(i+3,k);\
    a4 = alpha*A(i+4,k);\
    a5 = alpha*A(i+5,k);\
    a6 = alpha*A(i+6,k);\
    a7 = alpha*A(i+7,k);\
    b0 = B(k,j);\
    b1 = B(k,j+1);\
    b2 = B(k,j+2);\
    b3 = B(k,j+3);\
    b4 = B(k,j+4);\
    b5 = B(k,j+5);\
    b6 = B(k,j+6);\
    b7 = B(k,j+7);\
    c00 += wierdMul(a0, b0);\
    c01 += wierdMul(a0, b1);\
    c02 += wierdMul(a0, b2);\
    c03 += wierdMul(a0, b3);\
    c04 += wierdMul(a0, b4);\
    c05 += wierdMul(a0, b5);\
    c06 += wierdMul(a0, b6);\
    c07 += wierdMul(a0, b7);\
    c10 += wierdMul(a1, b0);\
    c11 += wierdMul(a1, b1);\
    c12 += wierdMul(a1, b2);\
    c13 += wierdMul(a1, b3);\
    c14 += wierdMul(a1, b4);\
    c15 += wierdMul(a1, b5);\
    c16 += wierdMul(a1, b6);\
    c17 += wierdMul(a1, b7);\
    c20 += wierdMul(a2, b0);\
    c21 += wierdMul(a2, b1);\
    c22 += wierdMul(a2, b2);\
    c23 += wierdMul(a2, b3);\
    c24 += wierdMul(a2, b4);\
    c25 += wierdMul(a2, b5);\
    c26 += wierdMul(a2, b6);\
    c27 += wierdMul(a2, b7);\
    c30 += wierdMul(a3, b0);\
    c31 += wierdMul(a3, b1);\
    c32 += wierdMul(a3, b2);\
    c33 += wierdMul(a3, b3);\
    c34 += wierdMul(a3, b4);\
    c35 += wierdMul(a3, b5);\
    c36 += wierdMul(a3, b6);\
    c37 += wierdMul(a3, b7);\
    c40 += wierdMul(a4, b0);\
    c41 += wierdMul(a4, b1);\
    c42 += wierdMul(a4, b2);\
    c43 += wierdMul(a4, b3);\
    c44 += wierdMul(a4, b4);\
    c45 += wierdMul(a4, b5);\
    c46 += wierdMul(a4, b6);\
    c47 += wierdMul(a4, b7);\
    c50 += wierdMul(a5, b0);\
    c51 += wierdMul(a5, b1);\
    c52 += wierdMul(a5, b2);\
    c53 += wierdMul(a5, b3);\
    c54 += wierdMul(a5, b4);\
    c55 += wierdMul(a5, b5);\
    c56 += wierdMul(a5, b6);\
    c57 += wierdMul(a5, b7);\
    c60 += wierdMul(a6, b0);\
    c61 += wierdMul(a6, b1);\
    c62 += wierdMul(a6, b2);\
    c63 += wierdMul(a6, b3);\
    c64 += wierdMul(a6, b4);\
    c65 += wierdMul(a6, b5);\
    c66 += wierdMul(a6, b6);\
    c67 += wierdMul(a6, b7);\
    c70 += wierdMul(a7, b0);\
    c71 += wierdMul(a7, b1);\
    c72 += wierdMul(a7, b2);\
    c73 += wierdMul(a7, b3);\
    c74 += wierdMul(a7, b4);\
    c75 += wierdMul(a7, b5);\
    c76 += wierdMul(a7, b6);\
    c77 += wierdMul(a7, b7);\
    k++;

void scale_c_k9(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_opt_k9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k9(C,M,N,LDC,beta);
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

void macro_dummy_cpu_8x8_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC){
    int i,j,k;
    // if (beta != 1.0) scale_c_k9(C,M,N,LDC,beta);
    int M8=M&-8,N8=N&-8,K4 = K & -4;
    for (i=0;i<M8;i+=8){
        for (j=0;j<N8;j+=8){
            double c00=C(i,j);
            double c01=C(i,j+1);
            double c02=C(i,j+2);
            double c03=C(i,j+3);
            double c04=C(i,j+4);
            double c05=C(i,j+5);
            double c06=C(i,j+6);
            double c07=C(i,j+7);
            double c10=C(i+1,j);
            double c11=C(i+1,j+1);
            double c12=C(i+1,j+2);
            double c13=C(i+1,j+3);
            double c14=C(i+1,j+4);
            double c15=C(i+1,j+5);
            double c16=C(i+1,j+6);
            double c17=C(i+1,j+7);
            double c20=C(i+2,j);
            double c21=C(i+2,j+1);
            double c22=C(i+2,j+2);
            double c23=C(i+2,j+3);
            double c24=C(i+2,j+4);
            double c25=C(i+2,j+5);
            double c26=C(i+2,j+6);
            double c27=C(i+2,j+7);
            double c30=C(i+3,j);
            double c31=C(i+3,j+1);
            double c32=C(i+3,j+2);
            double c33=C(i+3,j+3);
            double c34=C(i+3,j+4);
            double c35=C(i+3,j+5);
            double c36=C(i+3,j+6);
            double c37=C(i+3,j+7);
            double c40=C(i+4,j);
            double c41=C(i+4,j+1);
            double c42=C(i+4,j+2);
            double c43=C(i+4,j+3);
            double c44=C(i+4,j+4);
            double c45=C(i+4,j+5);
            double c46=C(i+4,j+6);
            double c47=C(i+4,j+7);
            double c50=C(i+5,j);
            double c51=C(i+5,j+1);
            double c52=C(i+5,j+2);
            double c53=C(i+5,j+3);
            double c54=C(i+5,j+4);
            double c55=C(i+5,j+5);
            double c56=C(i+5,j+6);
            double c57=C(i+5,j+7);
            double c60=C(i+6,j);
            double c61=C(i+6,j+1);
            double c62=C(i+6,j+2);
            double c63=C(i+6,j+3);
            double c64=C(i+6,j+4);
            double c65=C(i+6,j+5);
            double c66=C(i+6,j+6);
            double c67=C(i+6,j+7);
            double c70=C(i+7,j);
            double c71=C(i+7,j+1);
            double c72=C(i+7,j+2);
            double c73=C(i+7,j+3);
            double c74=C(i+7,j+4);
            double c75=C(i+7,j+5);
            double c76=C(i+7,j+6);
            double c77=C(i+7,j+7);
            double a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
            for (k=0;k<K4;){
                KERNEL_K9_8x8_naive;
                KERNEL_K9_8x8_naive;
                KERNEL_K9_8x8_naive;
                KERNEL_K9_8x8_naive;
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i,j+2) = c02;
            C(i,j+3) = c03;
            C(i,j+4) = c04;
            C(i,j+5) = c05;
            C(i,j+6) = c06;
            C(i,j+7) = c07;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
            C(i+1,j+2) = c12;
            C(i+1,j+3) = c13;
            C(i+1,j+4) = c14;
            C(i+1,j+5) = c15;
            C(i+1,j+6) = c16;
            C(i+1,j+7) = c17;
            C(i+2,j) = c20;
            C(i+2,j+1) = c21;
            C(i+2,j+2) = c22;
            C(i+2,j+3) = c23;
            C(i+2,j+4) = c24;
            C(i+2,j+5) = c25;
            C(i+2,j+6) = c26;
            C(i+2,j+7) = c27;
            C(i+3,j) = c30;
            C(i+3,j+1) = c31;
            C(i+3,j+2) = c32;
            C(i+3,j+3) = c33;
            C(i+3,j+4) = c34;
            C(i+3,j+5) = c35;
            C(i+3,j+6) = c36;
            C(i+3,j+7) = c37;
            C(i+4,j) = c40;
            C(i+4,j+1) = c41;
            C(i+4,j+2) = c42;
            C(i+4,j+3) = c43;
            C(i+4,j+4) = c44;
            C(i+4,j+5) = c45;
            C(i+4,j+6) = c46;
            C(i+4,j+7) = c47;
            C(i+5,j) = c50;
            C(i+5,j+1) = c51;
            C(i+5,j+2) = c52;
            C(i+5,j+3) = c53;
            C(i+5,j+4) = c54;
            C(i+5,j+5) = c55;
            C(i+5,j+6) = c56;
            C(i+5,j+7) = c57;
            C(i+6,j) = c60;
            C(i+6,j+1) = c61;
            C(i+6,j+2) = c62;
            C(i+6,j+3) = c63;
            C(i+6,j+4) = c64;
            C(i+6,j+5) = c65;
            C(i+6,j+6) = c66;
            C(i+6,j+7) = c67;
            C(i+7,j) = c70;
            C(i+7,j+1) = c71;
            C(i+7,j+2) = c72;
            C(i+7,j+3) = c73;
            C(i+7,j+4) = c74;
            C(i+7,j+5) = c75;
            C(i+7,j+6) = c76;
            C(i+7,j+7) = c77;
        }
    }
    if (M8==M&&N8==N) return;
    // boundary conditions
    if (M8!=M) mydgemm_cpu_opt_k9(M-M8,N,K,alpha,A+M8,LDA,B,LDB,1.0,&C(M8,0),LDC);
    if (N8!=N) mydgemm_cpu_opt_k9(M8,N-N8,K,alpha,A,LDA,&B(0,N8),LDB,1.0,&C(0,N8),LDC);
}

void mydgemm_cpu_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    
    if (beta != 1.0) scale_c_k9(C,M,N,LDC,beta);
    
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                //macro kernel: to compute C += A_tilt * B_tilt
                macro_dummy_cpu_8x8_v9(m_inc,n_inc,k_inc,alpha,&A(m_count,k_count), LDA, &B(k_count,n_count), LDB, &C(m_count, n_count), LDC);
            }
        }
    }

}

#endif