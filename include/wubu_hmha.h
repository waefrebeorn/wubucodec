/* GAP-C033: hyperbolic multi-head attention */
#ifndef WUBU_HMHA_H
#define WUBU_HMMA_H
#ifdef __cplusplus
extern "C" {
#endif
/* Wq/Wk/Wv/Wo: [D,D]. x/out: [N,D] on-ball. heads must divide D. */
int wubu_hmha_forward(const float* Wq,const float* Wk,const float* Wv,
                      const float* Wo,
                      const float* x,int N,int D,int heads,
                      float tau,float c,float* out);
#ifdef __cplusplus
}
#endif
#endif
