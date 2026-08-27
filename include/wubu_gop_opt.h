/* GAP-C061: adaptive GOP length */
#ifndef WUBU_GOP_OPT_H
#define WUBU_GOP_OPT_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_go_encode(const uint8_t* frames,int n_frames,
                     int W,int H,float angle_step,
                     float quality_thresh,int max_gop,FILE* out);
#ifdef __cplusplus
}
#endif
#endif
