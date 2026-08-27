/* Full H.264-style deblocking filter */
#ifndef WUBU_DEBLOCK2_H
#define WUBU_DEBLOCK2_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int  wubu_bs_compute(int is_intra_edge,int has_residual_diff,int mv_different);
int  wubu_deblock_decision(const void* ep,int qp);
void wubu_deblock_h264(uint8_t* img,int W,int H,int qp,int bs_threshold);
#ifdef __cplusplus
}
#endif
#endif
