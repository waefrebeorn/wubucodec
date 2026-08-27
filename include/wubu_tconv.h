/* GAP-C048: hyperbolic 1D temporal convolution */
#ifndef WUBU_TCONV_H
#define WUBU_TCONV_H
#ifdef __cplusplus
extern "C" {
#endif
/* seq: [T,D] on-ball; kernel: [K] weights (NULL = uniform).
 * out: [T,D] same length (edge-padded). */
int wubu_tc_conv1d(const float* seq,int T,int D,int K,float c,
                    const float* kernel,float bias,
                    unsigned seed,float* out);
#ifdef __cplusplus
}
#endif
#endif
