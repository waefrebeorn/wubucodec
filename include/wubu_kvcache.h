/* GAP-C036: KV cache for hyperbolic autoregressive decoding */
#ifndef WUBU_KVCACHE_H
#define WUBU_KVCACHE_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int max_T,len,D;
    float c;
    float* keys; /* [max_T,D] on-ball */
    float* vals; /* [max_T,D] on-ball */
} WubuKVCache;

int  wubu_kvc_init(WubuKVCache* kc,int max_T,int D,float c);
void wubu_kvc_free(WubuKVCache* kc);
void wubu_kvc_reset(WubuKVCache* kc);
int  wubu_kvc_append(WubuKVCache* kc,const float* k_new,const float* v_new);
void wubu_kvc_decode(const WubuKVCache* kc,const float* q,
                     float tau,float* out);
#ifdef __cplusplus
}
#endif
#endif
