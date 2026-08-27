/* GAP-C054: bilinear subpixel rotation */
#ifndef WUBU_RD_SUBPIX_H
#define WUBU_RD_SUBPIX_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_sp_rotate(const uint8_t* src,uint8_t* dst,
                     int W,int H,float angle);
#ifdef __cplusplus
}
#endif
#endif
