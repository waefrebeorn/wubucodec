/* GAP-D032: hyperbolic UMAP-style layout */
#ifndef WUBU_HUMAP_H
#define WUBU_HUMAP_H
#ifdef __cplusplus
extern "C" {
#endif
/* W: [n,n] fuzzy symmetrized weights from k-NN graph. */
int  wubu_um_fuzzy_weights(const float* xs,const int* adj_idx,
                            const int* adj_ptr,int n,int D,float c,
                            float* W);
float wubu_um_cross_entropy(const float* Wh,const float* ys,
                             int n,int D,float c);
#ifdef __cplusplus
}
#endif
#endif
