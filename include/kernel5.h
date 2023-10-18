
#include <arm_neon.h>
#include "utils.h"

#define A(i,j) A[(i)*LDA+(j)]
#define B(i,j) B[(i)*LDB+(j)]
#define C(i,j) C[(i)*LDC+(j)]

void scale_c_k5(double *C, int M, int N, int LDC, double scalar) {
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C(i,j) *= scalar;
        }
    }
}

void mydgemm_cpu_opt_k5(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC) {
    int i, j, k;
    if (beta != 1.0) scale_c_k5(C, M, N, LDC, beta);
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double tmp = C(i, j);
            for (k = 0; k < K; k++) {
                tmp += alpha*wierdMul(A(i,k), B(k,j));
            }
            C(i, j) = tmp;
        }
    }
}

void mydgemm_cpu_v5(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC) {
    int i, j, k;
    if (beta != 1.0) scale_c_k5(C, M, N, LDC, beta);
    int M2 = M & -2, N2 = N & -2; // adjust for 128-bit granularity of NEON
    float64x2_t valpha = vdupq_n_f64(alpha); // broadcast alpha to a 128-bit vector
    for (i = 0; i < M2; i += 2) {
        for (j = 0; j < N2; j += 2) {
            float64x2_t c0 = vdupq_n_f64(0.0);
            float64x2_t c1 = vdupq_n_f64(0.0);
            for (k = 0; k < K; k++) {
                // float64x2_t a = vmulq_f64(valpha, vld1q_f64(&A(i, k)));
                float64x2_t a = vdupq_n_f64(A(i, k));
                float64x2_t b0 = vdupq_n_f64(B(k, j));
                float64x2_t b1 = vdupq_n_f64(B(k, j + 1));
                // calculate difference
                float64x2_t diff0 = vsubq_f64(a, b0);
                float64x2_t diff1 = vsubq_f64(a, b1);
                c0 = vmlaq_f64(c0, diff0, diff0);
                c1 = vmlaq_f64(c1, diff1, diff1);
            }
            vst1q_f64(&C(i, j), vaddq_f64(c0, vld1q_f64(&C(i, j))));
            vst1q_f64(&C(i, j + 1), vaddq_f64(c1, vld1q_f64(&C(i, j + 1))));
        }
    }
    if (M2 == M && N2 == N) return;
    // boundary conditions
    if (M2 != M) mydgemm_cpu_opt_k5(M - M2, N, K, alpha, A + M2, LDA, B, LDB, 1.0, &C(M2, 0), LDC);
    if (N2 != N) mydgemm_cpu_opt_k5(M2, N - N2, K, alpha, A, LDA, &B(0, N2), LDB, 1.0, &C(0, N2), LDC);
}
