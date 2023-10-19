#include <arm_neon.h>
#include "utils.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c_k6(double *C, int M, int N, int LDC, double scalar){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= scalar;
        }
    }
}

void mydgemm_cpu_opt_k6(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k6(C,M,N,LDC,beta);
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

#define KERNEL_K1_2x4_neon_intrinsics\
    a = vmulq_f64(valpha, vld1q_f64(&A(i,k)));\
    b0 = vdupq_n_f64(B(k,j));\
    b1 = vdupq_n_f64(B(k,j+1));\
    b2 = vdupq_n_f64(B(k,j+2));\
    b3 = vdupq_n_f64(B(k,j+3));\
    diff0 = vsubq_f64(a, b0);\
    diff1 = vsubq_f64(a, b1);\
    diff2 = vsubq_f64(a, b2);\
    diff3 = vsubq_f64(a, b3);\
    c0 = vmlaq_f64(c0, diff0, diff0);\
    c1 = vmlaq_f64(c1, diff1, diff1);\
    c2 = vmlaq_f64(c2, diff2, diff2);\
    c3 = vmlaq_f64(c3, diff3, diff3);\
    k++;

void mydgemm_cpu_v6(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) scale_c_k6(C, M, N, LDC, beta);
    int M2 = M & -2, N4 = N & -4, K4 = K & -4;
    float64x2_t valpha = vdupq_n_f64(alpha);
    float64x2_t a, b0, b1, b2, b3, diff0, diff1, diff2, diff3;
    for (i = 0; i < M2; i+=2){
        for (j = 0; j < N4; j+=4){
            float64x2_t c0 = vdupq_n_f64(0);
            float64x2_t c1 = vdupq_n_f64(0);
            float64x2_t c2 = vdupq_n_f64(0);
            float64x2_t c3 = vdupq_n_f64(0);
            for (k = 0; k < K4;){
                KERNEL_K1_2x4_neon_intrinsics
                KERNEL_K1_2x4_neon_intrinsics
                KERNEL_K1_2x4_neon_intrinsics
                KERNEL_K1_2x4_neon_intrinsics
            }
            for (k = K4; k < K;){
                KERNEL_K1_2x4_neon_intrinsics
            }
            vst1q_f64(&C(i,j), vaddq_f64(c0, vld1q_f64(&C(i,j))));
            vst1q_f64(&C(i,j+1), vaddq_f64(c1, vld1q_f64(&C(i,j+1))));
            vst1q_f64(&C(i,j+2), vaddq_f64(c2, vld1q_f64(&C(i,j+2))));
            vst1q_f64(&C(i,j+3), vaddq_f64(c3, vld1q_f64(&C(i,j+3))));
        }
    }
    if (M2 == M && N4 == N) return;
    if (M2 != M) mydgemm_cpu_opt_k6(M-M2, N, K, alpha, A+M2, LDA, B, LDB, 1.0, &C(M2,0), LDC);
    if (N4 != N) mydgemm_cpu_opt_k6(M2, N-N4, K, alpha, A, LDA, &B(0,N4), LDB, 1.0, &C(0,N4), LDC);
}
