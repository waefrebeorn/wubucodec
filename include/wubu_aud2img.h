/* GAP-E003: audio→image reversible codec */
#ifndef WUBU_AUD2IMG_H
#define WUBU_AUD2IMG_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int  wubu_ai_encode(const float* audio,int n_samples,int fft_size,
                     int hop,int img_w,int img_h,uint8_t* img);
float wubu_ai_decode(const uint8_t* img,int img_w,int img_h,
                      int fft_size,int hop,float* audio,int max_samples);
float wubu_ai_correlation(const float* a,const float* b,int n);
#ifdef __cplusplus
}
#endif
#endif
