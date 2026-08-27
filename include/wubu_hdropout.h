/* GAP-C045: hyperbolic dropout (tangent-space) */
#ifndef WUBU_HDROPOUT_H
#define WUBU_HDROPOUT_H
#ifdef __cplusplus
extern "C" {
#endif
int wubu_hd_apply(const float* x,int D,float c,float p_rate,
                   unsigned* seed,float* out);
#ifdef __cplusplus
}
#endif
#endif
