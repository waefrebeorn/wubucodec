/* GAP-C043: spectral normalization (power iteration) */
#ifndef WUBU_SPECNORM_H
#define WUBU_SPECNORM_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_sn_estimate(const float* W,int rows,int cols,
                        int iters,unsigned* seed);
void wubu_sn_normalize(float* W,int rows,int cols,float sigma_max,
                        int iters,unsigned* seed);
#ifdef __cplusplus
}
#endif
#endif
