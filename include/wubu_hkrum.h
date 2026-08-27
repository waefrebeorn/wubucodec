/* GAP-D018: hyperbolic Krum outlier detection */
#ifndef WUBU_HKRUM_H
#define WUBU_HKRUM_H
#ifdef __cplusplus
extern "C" {
#endif
int  wubu_hkrum_scores(const float* pts,int n,int D,float c,int f,
                       float* scores);
int  wubu_hkrum_detect(const float* pts,int n,int D,float c,int f,
                        float multiplier,int* flags);
#ifdef __cplusplus
}
#endif
#endif
