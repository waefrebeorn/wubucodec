/* GAP-C017: Hyperboloid linear layer (HGCN Eq. 6-7) */
#ifndef WUBU_HLINEAR_H
#define WUBU_HLINEAR_H
#ifdef __cplusplus
extern "C" {
#endif
/* x: [N, D_in+1] points on the Lorentz hyperboloid (curvature -c).
 * W: [D_out, D_in] tangent-space weight matrix.
 * b_space: [D_out] bias in origin's tangent frame, or NULL.
 * out: [N, D_out+1] — result on the hyperboloid. Returns 0 on success. */
int wubu_hlinear_forward(const float* W,const float* b_space,
                         const float* x,int N,int D_in,int D_out,
                         float c,float* out);
/* GAP-C018: hyperbolic activation — log0 -> pointwise act -> exp0.
 * act_name: 0=ReLU, 1=tanh. x/out: [N, D+1] on the hyperboloid. */
int wubu_hactivation(const float* x,int N,int D,float c,
                     int act_name,float* out);
#ifdef __cplusplus
}
#endif
#endif
