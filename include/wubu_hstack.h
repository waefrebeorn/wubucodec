/* GAP-C040: multi-layer hyperbolic GCN stack with residuals */
#ifndef WUBU_HSTACK_H
#define WUBU_HSTACK_H
#ifdef __cplusplus
extern "C" {
#endif
/* W: [L, D, D] per-layer weights. alpha: residual blend (0=no update). */
int wubu_hstack_forward(const float* x,const int* adj_idx,
                         const int* adj_ptr,const float* edge_weight,
                         const float* W,int L,int N,int D,float c,
                         float alpha,float* out);
float wubu_hstack_distinctness(const float* embs,int n,int D,float c);
#ifdef __cplusplus
}
#endif
#endif
