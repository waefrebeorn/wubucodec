/* GROUP 1: Sub-pixel motion estimation */
#ifndef WUBU_SUBPEL_H
#define WUBU_SUBPEL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void   wubu_sp_halfpel(const uint8_t* src,uint8_t* dst,int W,int H);
uint8_t wubu_sp_get(const uint8_t* hp,int W2,int H2,
                     int x_int,int y_int,int dx,int dy);
long   wubu_sp_me(const uint8_t* curr,const uint8_t* hp_ref,
                   int W2,int H2,int bx,int by,int bs,
                   int search_range,int* out_dx,int* out_dy);
#ifdef __cplusplus
}
#endif
#endif
