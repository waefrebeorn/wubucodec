/* GROUP 11: Sample Adaptive Offset */
#ifndef WUBU_SAO_H
#define WUBU_SAO_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_sao_band(const uint8_t* img,uint8_t* output,int W,int H,
                     int band_pos,const int* band_offsets,int num_bands);
void wubu_sao_edge(const uint8_t* img,uint8_t* output,int W,int H,
                     int dir,const int* eo_offsets);
void wubu_sao_band_estimate(const uint8_t* orig,const uint8_t* recon,
                              int W,int H,int band_pos,
                              int* out_offsets,int num_bands);
void wubu_sao_edge_estimate(const uint8_t* orig,const uint8_t* recon,
                              int W,int H,int dir,int* out_offsets);
#ifdef __cplusplus
}
#endif
#endif
