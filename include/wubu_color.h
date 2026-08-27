/* GROUP 17: Color space pipeline */
#ifndef WUBU_COLOR_H
#define WUBU_COLOR_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    WUBU_CS_BT601,
    WUBU_CS_BT709,
    WUBU_CS_BT2020
} wubu_color_std_t;

void wubu_rgb_to_yuv_cm(const uint8_t* rgb,uint8_t* y,uint8_t* cb,uint8_t* cr,
                          long n_pixels,wubu_color_std_t std);
void wubu_rgb_to_709(const uint8_t* rgb,uint8_t* y,uint8_t* u,uint8_t* v,
                      long n_pixels);
void wubu_rgb_to_2020(const uint8_t* rgb,uint16_t* y,uint16_t* u,uint16_t* v,
                       long n_pixels);
void wubu_tonemap_reinhard(const float* hdr_rgb,uint8_t* sdr_rgb,
                             long n_pixels,float max_luminance);
float wubu_pq_eotf(uint16_t pq_value);
#ifdef __cplusplus
}
#endif
#endif
