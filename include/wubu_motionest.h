/* GAP-C075: block matching motion estimation */
#ifndef WUBU_MOTIONEST_H
#define WUBU_MOTIONEST_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_me_sad(const uint8_t*,const uint8_t*,int,int,int,int,int,int,int);
long wubu_me_block(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,
                    int search_range,int* out_dx,int* out_dy);
int  wubu_me_frame(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bs,int search_range,int* out_mvs);
void wubu_me_compensate(const uint8_t* ref,int W,int H,int bs,
                          const int* mvs,uint8_t* predicted);
#ifdef __cplusplus
}
#endif
#endif
