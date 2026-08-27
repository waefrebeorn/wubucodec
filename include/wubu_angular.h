/* GROUP 9: Angular intra prediction (35 modes) */
#ifndef WUBU_ANGULAR_H
#define WUBU_ANGULAR_H
#include <stdint.h>
#define BS_MAX 32
#ifdef __cplusplus
extern "C" {
#endif
void wubu_ipred(const uint8_t* img,int W,int H,
                 int bx,int by,int bs,int mode,uint8_t* block);
int  wubu_ipred_best_mode(const uint8_t* img,int W,int H,
                           int bx,int by,int bs,const uint8_t* actual);
#ifdef __cplusplus
}
#endif
#endif
