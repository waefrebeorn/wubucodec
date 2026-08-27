/* GAP-C053: rate-distortion curve generator */
#ifndef WUBU_RD_CURVE_H
#define WUBU_RD_CURVE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_rd_gen_rotation(uint8_t* frames,int W,int H,int NF,float angle_step);
long wubu_rd_encode_euclid(const uint8_t* frame,const uint8_t* prev,
                            int W,int H,int qshift,uint8_t* recon);
long wubu_rd_encode_quat(const uint8_t* prev,const uint8_t* curr,
                          float true_angle,int W,int H,
                          int angle_bits,uint8_t* recon);
#ifdef __cplusplus
}
#endif
#endif
