/* GAP-C041: hyperbolic graph attention layer */
#ifndef WUBU_HGAT_H
#define WUBU_HGAT_H
#ifdef __cplusplus
extern "C" {
#endif
/* distance-weighted attention aggregation with mobius residual.
 * alpha_vec: [D] learned per-dim scale or NULL. */
int wubu_hgat_forward(const float* x,const int* adj_idx,
                       const int* adj_ptr,const float* W_self,
                       const float* alpha_vec,int N,int D,float c,
                       float tau,float* out);
#ifdef __cplusplus
}
#endif
#endif
