/* GAP-A007: k-NN graph construction on the Poincaré ball */
#ifndef WUBU_KNNGRAPH_H
#define WUBU_KNNGRAPH_H
#ifdef __cplusplus
extern "C" {
#endif
/* builds CSR adjacency; mutual=1 keeps only bidirectional edges.
 * Caller frees *out_idx and *out_ptr. */
int wubu_knng_build(const float* pts,int n,int D,int k,float c,int mutual,
                    int** out_idx,int** out_ptr,int* out_nnz);
#ifdef __cplusplus
}
#endif
#endif
