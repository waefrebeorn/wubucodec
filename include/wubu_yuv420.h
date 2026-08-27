/* GAP-C076: RGB↔YUV420 conversion */
#ifndef WUBU_YUV420_H
#define WUBU_YUV420_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_rgb_to_yuv420(const uint8_t* rgb,uint8_t* y,uint8_t* u,uint8_t* v,
                          int W,int H);
void wubu_yuv420_to_rgb(const uint8_t* y,const uint8_t* u,const uint8_t* v,
                          uint8_t* rgb,int W,int H);
long wubu_yuv420_size(int W,int H);
#ifdef __cplusplus
}
#endif
#endif
