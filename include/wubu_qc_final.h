/* GAP-C058: final quaternion codec with SLERP prediction */
#ifndef WUBU_QC_FINAL_H
#define WUBU_QC_FINAL_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_qf_encode(const uint8_t* frames,int n_frames,
                     int W,int H,float true_angle_step,FILE* out);
void wubu_qf_decode(FILE* in,uint8_t* frames_out,
                     int n_frames,int W,int H,float angle_step);
#ifdef __cplusplus
}
#endif
#endif
