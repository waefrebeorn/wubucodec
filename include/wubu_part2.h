/* GROUP 1: remaining gaps */
#ifndef WUBU_PART2_H
#define WUBU_PART2_H
#include <stdint.h>
#include "wubu_partitions.h"
#ifdef __cplusplus
extern "C" {
#endif
int    wubu_amp_partitions(int block_size,wubu_partition_t* parts);
void   wubu_mv_round_halfpel(int* dx,int* dy);
void   wubu_mv_round_integer(int* dx,int* dy);
int    wubu_adapt_search_range(const int16_t* mv_field,int blocks_per_row,
                                 int block_idx,int base_range);
long   wubu_sad_aligned(const uint8_t* a,const uint8_t* b,int n);
double wubu_rd_cost(long sad,int mvd_x,int mvd_y,double lambda);
int    wubu_rd_best_mv(const long* sads,const int* dxs,const int* dys,
                         int n_candidates,double lambda,int* out_idx);
#ifdef __cplusplus
}
#endif
#endif
