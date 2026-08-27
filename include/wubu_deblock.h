/* GAP-C073: deblocking filter */
#ifndef WUBU_DEBLOCK_H
#define WUBU_DEBLOCK_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_db_filter(uint8_t* img,int W,int H,int quality);
void wubu_db_filter_plane(uint8_t* img,int W,int H,int quality);
#ifdef __cplusplus
}
#endif
#endif
