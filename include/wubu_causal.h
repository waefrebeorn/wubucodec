/* GAP-C035: causal masking for autoregressive attention */
#ifndef WUBU_CAUSAL_H
#define WUBU_CAUSAL_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_causal_apply(float* attn,int N);
void wubu_causal_mask(int N,uint8_t* mask);
int  wubu_causal_respected(const float* attn,int N,float eps);
#ifdef __cplusplus
}
#endif
#endif
