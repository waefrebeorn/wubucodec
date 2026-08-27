/* GAP-D026: hyperbolic t-SNE (CO-SNE simplified) */
#ifndef WUBU_HTSNE_H
#define WUBU_HTSNE_H
#ifdef __cplusplus
extern "C" {
#endif
/* P: [n,n] high-dim gaussian affinities (median-heuristic sigma). */
int  wubu_htsne_p_high(const float* xs,int n,int D_in,float* P);
void wubu_htsne_q_low(const float* ys,int n,int D,float c,float* Q);
float wubu_htsne_kl(const float* P,const float* Q,int n);
#ifdef __cplusplus
}
#endif
#endif
