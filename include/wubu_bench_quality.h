/* GAP-C052: PSNR + SSIM + MAE quality metrics */
#ifndef WUBU_BENCH_QUALITY_H
#define WUBU_BENCH_QUALITY_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
float wubu_q_psnr(const uint8_t* orig,const uint8_t* recon,long n_pixels);
float wubu_q_ssim(const uint8_t* a,const uint8_t* b,long n);
float wubu_q_mae(const uint8_t* orig,const uint8_t* recon,long n_pixels);
void  wubu_q_report(const uint8_t* orig,const uint8_t* recon,
                     int W,int H,float* psnr,float* ssim,float* mae);
#ifdef __cplusplus
}
#endif
#endif
