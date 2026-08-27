/* GROUP 20: Video quality metrics */
#ifndef WUBU_SSIM_H
#define WUBU_SSIM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
double wubu_psnr_channel(const uint8_t* a,const uint8_t* b,long n);
double wubu_ssim(const uint8_t* x,const uint8_t* y,int W,int H);
double wubu_msssim(const uint8_t* x,const uint8_t* y,int W,int H,int n_scales);
double wubu_bdrate(const double* rate_a,const double* psnr_a,
                    const double* rate_b,const double* psnr_b,int n_points);
#ifdef __cplusplus
}
#endif
#endif
