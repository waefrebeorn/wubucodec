/* GAP-B019: attention Frobenius norm monitor */
#ifndef WUBU_ATTNFROBENIUS_H
#define WUBU_ATTNFROBENIUS_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_af_frobenius(const float* attn,int N,int n_heads);
/* 1.0 = all rows uniform (healthy), 0.0 = all one-hot (collapsed) */
float wubu_af_uniformity(const float* attn,int N,int n_heads);
#ifdef __cplusplus
}
#endif
#endif
