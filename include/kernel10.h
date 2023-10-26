#ifndef _KNL10_H_
#define _KNL10_H_

#include <arm_neon.h>
#include "utils.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

#define M_BLOCKING 192
#define N_BLOCKING 2048
#define K_BLOCKING 384

void scale_c_k10(double *C, int M, int N, int LDC, double scalar){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= scalar;
        }
    }
}

void packing_a_k10(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    for (count_first=0;count_sub>3;count_first+=4,count_sub-=4){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            // pack 4 elements a time
            vst1q_f64(todst,vld1q_f64(tosrc));
            tosrc += 2;
            todst+=2;
            vst1q_f64(todst,vld1q_f64(tosrc));
            tosrc+= - 2 + leading_dim;
            todst+=2;
        }
    }
}

void packing_b_k10(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*todst;
    todst=dst;
    int count_first,count_second;
    for (count_second=0;count_second<dim_second;count_second+=4){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
            *todst=*tosrc3;tosrc3++;todst++;
            *todst=*tosrc4;tosrc4++;todst++;
        }
    }
}

void mydgemm_cpu_opt_k10(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
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

#define macro_kernel_4xkx4_packing\
    c00 = vdupq_n_f64(0);\
    c01 = vdupq_n_f64(0);\
    c02 = vdupq_n_f64(0);\
    c03 = vdupq_n_f64(0);\
    c10 = vdupq_n_f64(0);\
    c11 = vdupq_n_f64(0);\
    c12 = vdupq_n_f64(0);\
    c13 = vdupq_n_f64(0);\
    for (k = 0; k < K4;){\
        KERNEL_K10_4x4_neon_intrinsics\
        KERNEL_K10_4x4_neon_intrinsics\
        KERNEL_K10_4x4_neon_intrinsics\
        KERNEL_K10_4x4_neon_intrinsics\
    }\
    for (k = K4; k < K;){\
        KERNEL_K10_4x4_neon_intrinsics\
    }\
    vst1q_f64(&C(i,j), vaddq_f64(c00, vld1q_f64(&C(i,j))));\
    vst1q_f64(&C(i,j+1), vaddq_f64(c01, vld1q_f64(&C(i,j+1))));\
    vst1q_f64(&C(i,j+2), vaddq_f64(c02, vld1q_f64(&C(i,j+2))));\
    vst1q_f64(&C(i,j+3), vaddq_f64(c03, vld1q_f64(&C(i,j+3))));\
    vst1q_f64(&C(i+2,j), vaddq_f64(c10, vld1q_f64(&C(i+2,j))));\
    vst1q_f64(&C(i+2,j+1), vaddq_f64(c11, vld1q_f64(&C(i+2,j+1))));\
    vst1q_f64(&C(i+2,j+2), vaddq_f64(c12, vld1q_f64(&C(i+2,j+2))));\
    vst1q_f64(&C(i+2,j+3), vaddq_f64(c13, vld1q_f64(&C(i+2,j+3))));

#define KERNEL_K10_4x4_neon_intrinsics\
    a1 = vmulq_f64(valpha, vld1q_f64(ptr_packing_a));\
    a2 = vmulq_f64(valpha, vld1q_f64(ptr_packing_a+2));\
    b0 = vdupq_n_f64(*ptr_packing_b);\
    b1 = vdupq_n_f64(*(ptr_packing_b+1));\
    b2 = vdupq_n_f64(*(ptr_packing_b+2));\
    b3 = vdupq_n_f64(*(ptr_packing_b+3));\
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
    ptr_packing_a+=4;ptr_packing_b+=4;k++;

void macro_kernel_gemm_k10(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC){
    int i,j,k;
    int M4=M&-4,N4=N&-4,K4=K&-4;
    double *ptr_packing_a = A;
    double *ptr_packing_b = B;
    float64x2_t valpha = vdupq_n_f64(alpha); //broadcast alpha to a 128-bit vector
    float64x2_t a1, a2, b0, b1, b2, b3, diff00, diff01, diff02, diff03, diff10, diff11, diff12, diff13;
    float64x2_t c00, c01, c02, c03, c10, c11, c12, c13;
    for (i=0;i<M4;i+=4){
        for (j=0;j<N4;j+=4){
            ptr_packing_a=A+i*K;ptr_packing_b=B+j*K;
            macro_kernel_4xkx4_packing
        }
    }
}

void mydgemm_cpu_v10(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    
    if (beta != 1.0) scale_c_k10(C,M,N,LDC,beta);
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            packing_b_k10(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                packing_a_k10(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                //macro kernel: to compute C += A_tilt * B_tilt
                macro_kernel_gemm_k10(m_inc,n_inc,k_inc,alpha,a_buffer, LDA, b_buffer, LDB, &C(m_count, n_count), LDC);
            }
        }
    }
    free(a_buffer);free(b_buffer);
}

#endif // _KNL10_H_