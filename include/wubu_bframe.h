/* GAP-C060: B-frame bidirectional prediction for quaternion codec */
#ifndef WUBU_BFRAME_H
#define WUBU_BFRAME_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_bf_encode(const uint8_t* frames,int n_frames,
                     int W,int H,float angle_step,FILE* out);
#ifdef __cplusplus
}
#endif
#endif
