/* SIMD-accelerated operations for WUBQ codec hot paths */
#include <stdint.h>
#include <stdlib.h>
#ifndef WUBU_SIMD_H
#define WUBU_SIMD_H
#ifdef __cplusplus
extern "C" {
#endif

void wubu_simd_hamilton(const float* a,const float* b,float* out);
void wubu_simd_hamilton_batch(const float* a,const float* b,float* out,int n);
long wubu_simd_sad(const uint8_t* a,const uint8_t* b,int n);
void wubu_simd_normalize(float* q);

#ifdef __cplusplus
}
#endif
#endif
