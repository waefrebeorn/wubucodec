/* SIMD DCT */
#ifndef WUBU_SIMD_DCT_H
#define WUBU_SIMD_DCT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_sdct_scalar(const int16_t* block,int16_t* output);
void wubu_sdct_avx2(const int16_t* block,int16_t* output);
long wubu_sdct_batch(const int16_t* blocks,int16_t* outputs,int n_blocks);
#ifdef __cplusplus
}
#endif
#endif
