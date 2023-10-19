#include <arm_neon.h>

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c_k2(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k2(C,M,N,LDC,beta);

    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            double tmp = C(i,j);
            int k_end = K & (~1);  // Largest even number less than or equal to K

            // // NEON optimized block
            // float64x2_t vec_alpha = vdupq_n_f64(alpha);
            // float64x2_t vec_tmp = vdupq_n_f64(0.0);
            // for (k=0; k < k_end; k += 2) {
            //     float64x2_t vec_a = vld1q_f64(&A(i,k));
            //     float64x2_t vec_b0 = vdupq_n_f64(B(k,j));
            //     float64x2_t vec_b1 = vdupq_n_f64(B(k+1,j));
            //     vec_tmp = vmlaq_f64(vec_tmp, vec_alpha, vec_a, vcombine_f64(vec_b0[0], vec_b1[0]));
            // }
            // tmp += vgetq_lane_f64(vec_tmp, 0) + vgetq_lane_f64(vec_tmp, 1);
            // NEON optimized block
            float64x2_t vec_alpha = vdupq_n_f64(alpha);
            float64x2_t vec_tmp = vdupq_n_f64(0.0);
            for (k=0; k < k_end; k += 2) {
                float64x2_t vec_a = vld1q_f64(&A(i,k));
                float64x1_t vec_b0 = vdup_n_f64(B(k,j));
                float64x1_t vec_b1 = vdup_n_f64(B(k+1,j));
                float64x2_t vec_b_combined = vcombine_f64(vec_b0, vec_b1);
                vec_tmp = vmlaq_f64(vec_tmp, vec_alpha, vec_a, vec_b_combined);
            }
            tmp += vgetq_lane_f64(vec_tmp, 0) + vgetq_lane_f64(vec_tmp, 1);


            // Handle any remaining elements
            for (; k < K; k++) {
                tmp += alpha * A(i,k) * B(k,j);
            }

            C(i,j) = tmp;
        }
    }
}
