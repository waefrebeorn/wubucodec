/* GROUP 13: Film Grain Synthesis */
#ifndef WUBU_FGS_H
#define WUBU_FGS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_fgs_estimate_ar(const int16_t* residual,long n_samples,
                            double* ar_coeffs);
void wubu_fgs_estimate_scaling(const uint8_t* source,const uint8_t* denoised,
                                 long n_pixels,int* scaling_lut);
void wubu_fgs_gen_template(const double* ar_coeffs,uint8_t* template_64x64,
                             unsigned int seed);
void wubu_fgs_apply(uint8_t* decoded,const uint8_t* template_64x64,
                      const int* scaling_lut,long W,long H,unsigned int frame_seed);
#ifdef __cplusplus
}
#endif
#endif
