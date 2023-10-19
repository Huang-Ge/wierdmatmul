#include <arm_neon.h>
#include "utils.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c_k7(double *C, int M, int N, int LDC, double scalar){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= scalar;
        }
    }
}

void mydgemm_cpu_opt_k7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k7(C,M,N,LDC,beta);
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

#define KERNEL_K1_4x4_neon_intrinsics\
    a1 = vmulq_f64(valpha, vld1q_f64(&A(i,k)));\
    a2 = vmulq_f64(valpha, vld1q_f64(&A(i+2,k)));\
    b0 = vdupq_n_f64(B(k,j));\
    b1 = vdupq_n_f64(B(k,j+1));\
    b2 = vdupq_n_f64(B(k,j+2));\
    b3 = vdupq_n_f64(B(k,j+3));\
    diff00 = vsubq_f64(a1, b0);\
    diff01 = vsubq_f64(a1, b1);\
    diff02 = vsubq_f64(a1, b2);\
    diff03 = vsubq_f64(a1, b3);\
    diff10 = vsubq_f64(a2, b0);\
    diff11 = vsubq_f64(a2, b1);\
    diff12 = vsubq_f64(a2, b2);\
    diff13 = vsubq_f64(a2, b3);\
    c00 = vmlaq_f64(c00, diff00, diff00);\
    c01 = vmlaq_f64(c01, diff01, diff01);\
    c02 = vmlaq_f64(c02, diff02, diff02);\
    c03 = vmlaq_f64(c03, diff03, diff03);\
    c10 = vmlaq_f64(c10, diff10, diff10);\
    c11 = vmlaq_f64(c11, diff11, diff11);\
    c12 = vmlaq_f64(c12, diff12, diff12);\
    c13 = vmlaq_f64(c13, diff13, diff13);\
    k++;

void mydgemm_cpu_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) scale_c_k7(C, M, N, LDC, beta);
    int M4 = M & -4, N4 = N & -4, K4 = K & -4;
    float64x2_t valpha = vdupq_n_f64(alpha);
    float64x2_t a1, a2, b0, b1, b2, b3, diff00, diff01, diff02, diff03, diff10, diff11, diff12, diff13;
    for (i = 0; i < M4; i+=4){
        for (j = 0; j < N4; j+=4){
            float64x2_t c00 = vdupq_n_f64(0);
            float64x2_t c01 = vdupq_n_f64(0);
            float64x2_t c02 = vdupq_n_f64(0);
            float64x2_t c03 = vdupq_n_f64(0);
            float64x2_t c10 = vdupq_n_f64(0);
            float64x2_t c11 = vdupq_n_f64(0);
            float64x2_t c12 = vdupq_n_f64(0);
            float64x2_t c13 = vdupq_n_f64(0);
            for (k = 0; k < K4;){
                KERNEL_K1_4x4_neon_intrinsics
                KERNEL_K1_4x4_neon_intrinsics
                KERNEL_K1_4x4_neon_intrinsics
                KERNEL_K1_4x4_neon_intrinsics
            }
            for (k = K4; k < K;){
                KERNEL_K1_4x4_neon_intrinsics
            }
            vst1q_f64(&C(i,j), vaddq_f64(c00, vld1q_f64(&C(i,j))));
            vst1q_f64(&C(i,j+1), vaddq_f64(c01, vld1q_f64(&C(i,j+1))));
            vst1q_f64(&C(i,j+2), vaddq_f64(c02, vld1q_f64(&C(i,j+2))));
            vst1q_f64(&C(i,j+3), vaddq_f64(c03, vld1q_f64(&C(i,j+3))));
            vst1q_f64(&C(i+2,j), vaddq_f64(c10, vld1q_f64(&C(i+2,j))));
            vst1q_f64(&C(i+2,j+1), vaddq_f64(c11, vld1q_f64(&C(i+2,j+1))));
            vst1q_f64(&C(i+2,j+2), vaddq_f64(c12, vld1q_f64(&C(i+2,j+2))));
            vst1q_f64(&C(i+2,j+3), vaddq_f64(c13, vld1q_f64(&C(i+2,j+3))));
        }
    }
    if (M4 == M && N4 == N) return;
    if (M4 != M) mydgemm_cpu_opt_k7(M-M4, N, K, alpha, A+M4, LDA, B, LDB, 1.0, &C(M4,0), LDC);
    if (N4 != N) mydgemm_cpu_opt_k7(M4, N-N4, K, alpha, A, LDA, &B(0,N4), LDB, 1.0, &C(0,N4), LDC);
}
