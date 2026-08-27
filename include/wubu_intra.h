/* GAP-C074: intra prediction modes */
#ifndef WUBU_INTRA_H
#define WUBU_INTRA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_ip_predict(uint8_t* img,int W,int H,int bx,int by,
                      int mode,uint8_t* block);
int  wubu_ip_best_mode(const uint8_t* img,const uint8_t* actual,
                        int W,int H,int bx,int by);
#ifdef __cplusplus
}
#endif
#endif
