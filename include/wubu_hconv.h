/* GAP-C027: hyperbolic pointwise convolution (HNN++ Poincaré MLP) */
#ifndef WUBU_HCONV_H
#define WUBU_HCONV_H
#ifdef __cplusplus
extern "C" {
#endif
int   wubu_hconv_forward(const float* W,const float* b,
                         const float* x,int N,int D,int D_out,
                         float c,float* out);
float wubu_hconv_bias_at(const float* b,int j);
#ifdef __cplusplus
}
#endif
#endif
