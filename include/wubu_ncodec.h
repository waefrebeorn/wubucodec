/* GROUP 25: Neural codec building blocks */
#ifndef WUBU_NCODEC_H
#define WUBU_NCODEC_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
double wubu_nc_logistic_cdf(double x,double mean,double scale);
double wubu_nc_prob_mass(int x_val,double scale);
double wubu_nc_bits_per_symbol(int x_val,double scale);
double wubu_nc_rate_estimate(const int16_t* latents,const double* scales,
                               long n_latents);
double wubu_nc_rd_loss(const float* original,const float* reconstructed,
                         const int16_t* latents,const double* scales,
                         long n_pixels,long n_latents,double lambda);
void   wubu_nc_hyperprior_scale(const int16_t* latents,long W,long H,
                                  double base_scale,double* out_scales);

typedef struct {
    uint64_t low;
    uint32_t range;
    uint8_t* buf;
    size_t cap;
    size_t bitpos;
} NCArithEnc;
void wubu_nc_arith_init(NCArithEnc* e,uint8_t* buf,size_t cap);
void wubu_nc_arith_encode(NCArithEnc* e,int bit,double prob_one);
long wubu_nc_arith_finish(NCArithEnc* e);
#ifdef __cplusplus
}
#endif
#endif
