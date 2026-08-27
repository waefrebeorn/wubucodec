/* GROUP 1: Skip mode + Merge mode + MV prediction */
#ifndef WUBU_MVPREP_H
#define WUBU_MVPREP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int  wubu_skip_detect(const uint8_t* curr,const uint8_t* predicted,
                       int W,int H,int bx,int by,int bs,long threshold);
int  wubu_merge_candidates(const int16_t* mv_field,uint8_t* avail_map,
                            int blocks_per_row,int block_idx,
                            int16_t* out_candidates);
void wubu_amvp_select(const int16_t* actual_mv,
                       const int16_t* candidates,int n_candidates,
                       int* best_idx,int16_t* mvd_out);
void wubu_amvp_reconstruct(const int16_t* mvp,const int16_t* mvd,
                            int16_t* mv_out);
#ifdef __cplusplus
}
#endif
#endif
