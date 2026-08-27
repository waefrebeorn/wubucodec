/* GAP-C046: hyperbolic gradient clipping */
#ifndef WUBU_HCLIP_H
#define WUBU_HCLIP_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_hc_riemannian_norm(const float* egrad,const float* x,
                               int D,float c);
/* returns 1 if clipped, 0 if untouched, -1 on bad args */
int  wubu_hc_clip(float* grad,const float* x,int D,float c,
                  float max_rnorm);
#ifdef __cplusplus
}
#endif
#endif
