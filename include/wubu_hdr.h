/* GROUP 24: HDR & Wide Color Gamut */
#ifndef WUBU_HDR_H
#define WUBU_HDR_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
double wubu_pq_eotf_full(int pq_value,int bit_depth);
int    wubu_pq_inverse_eotf(double nits,int bit_depth);
double wubu_hlg_oetf(double scene_linear);
double wubu_hlg_inverse(double signal);
void   wubu_hdr_compute_metadata(const uint16_t* luma,long n_pixels,int bit_depth,
                                   int* max_cll,int* max_fall);
void   wubu_hdr_to_sdr_tonemap(const uint16_t* hdr_luma,long n_pixels,
                                 int hdr_bit_depth,float hdr_max_nits,
                                 uint8_t* sdr_output);
#ifdef __cplusplus
}
#endif
#endif
