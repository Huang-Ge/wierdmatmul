# Matmul dummy project

## Problem Set

Implimentation of a variant of matrix multiplification: $C_{i,j} = \sum_k (A_{i, k} - B_{k,j})^2$.

## Description

This project is a onboarding project that tries to accrelarate a special way of matrix multiplication on Mac OS with M1-chip (arm archetecture). 

Instead of avx used on x86 Intel chips, M1 chip has arm architectured 128-bit register (256-bit in AVX) which can be accessed by using `#include <arm_neon.h>`.  In the local test environment, its performance is texted but the result shows that it is not achieving the performance boost as expected. While the reason is yet to be inspected, the test results are listed below.

Since the task is to accelarate the calculation as fast as possible locally, starting from kernel 8,  furthur ideas of matrix multiplication such as cache blocking are applied to the plain implementation.

## Performance

### Naive Approach

| Kernel # | GFLOPS       | Tricks                                            |
| -------- | ------------ | ------------------------------------------------- |
| 1        | 0.537377     | -                                                 |
| 2        | 0.551714     | Register reuse                                    |
| 3        | 0.786101     | 2x2 register blocking                             |
| 4        | 1.192514     | 4x4 register blocking                             |
| 8        | 1.184126     | 4 x 4 kernel + cache blocking                     |
| 9        | **1.409902** | 8 x 8 kernel +loop unrolling x 4 + cache blocking |

### NEON (AVX equivelent in arm)

| Kernel # | GFLOPS   | Tricks                                       |
| -------- | -------- | -------------------------------------------- |
| 5        | 0.439499 | 2x4 register blocking                        |
| 6        | 0.628501 | 2x4 register blocking + loop unrolling x 4   |
| 7        | 0.817736 | 4 x 4 kernel  + loop unrolling x 4           |
| 10       | 0.846954 | 4 x 4 kernel  + loop unrolling x 4 + packing |

## Kernel list

- Kernel 1
  - Most naive implementation of wierdMatMul, this kernel is used for correctiveness check.
  - Average performance: **0.537377 GFLOPS**
- Kernel 2
  - Register reuse before entering k-loop
  - Average performance: **0.551714 GFLOPS**
- Kernel 3
  - 2x2 register blocking
  - Average performance: **0.786101 GFLOPS**
- Kernel 4
  - 4x4 register blocking
  - Average performance: **1.192514 GFLOPS**
- Kernel 5
  - 2x4 register blocking with *NEON*,  but performance drastically dropped :(
  - Average performance: **0.439499 GFLOPS**
- Kernel 6
  - Kernel 5 + loop unrolling x 4, still low performance
  - Average performance: **0.628501 GFLOPS**
- Kernel 7
  - 4 x 4 kernel + NEON + loop unrolling x 4
  - Average performance: **0.817736 GFLOPS**
- Kernel 8
  - 4 x 4 kernel + cache blocking 
  - Average performance: **1.184126 GFLOPS** (which maintains when scaled up)
- Kernel 9
  - 8 x 8 kernel +loop unrolling x 4 + cache blocking 
  - Average performance: **1.409902 GFLOPS**
- Kernel 10
  - <u>Kernel 7</u> + packing. *Although NEON wasn't performing as expected, we are still gonna try packing out here*
  - Average performance: **0.846954 GFLOPS**
  - As a result, packing Still didn't do any better

#### Summary

The best I can get so far is 1.409902 GFLOPS. Implementation on NEON isn't working quit as efficient as expected. I stopped going further because it doesn't look like a promising road to accelerate matrix multiplication by imitating AVX using NEON.

### Reference

https://github.com/yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F#kernel-10-24x8-kernel--avx512--blocking--packing
