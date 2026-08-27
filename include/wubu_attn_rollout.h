/* GAP-B016: attention rollout */
#ifndef WUBU_ATTN_ROLLOUT_H
#define WUBU_ATTN_ROLLOUT_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_ar_layer(const float* attn,int N,float* out);
void wubu_ar_rollout(const float** attns,const int* num_layers_ptr,
                      int N,float* out);
void wubu_ar_distance_attention(const float* embs,int D,int N,
                                 float c,float tau,float* attn);
#ifdef __cplusplus
}
#endif
#endif
