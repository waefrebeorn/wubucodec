/* GAP-C022: Tangent-space-free Lorentz linear layer (SRBGCN/H2H-GCN) */
#ifndef WUBU_LDIRECT_H
#define WUBU_LDIRECT_H
#ifdef __cplusplus
extern "C" {
#endif
/* Direct hyperboloid linear: split time/space, W on space, recompute time.
 * x: [N, D_in+1], out: [N, D_out+1]. Returns 0 on success. */
int wubu_ldirect_forward(const float* W,const float* b_space,
                         const float* x,int N,int D_in,int D_out,
                         float c,float* out);
#ifdef __cplusplus
}
#endif
#endif
