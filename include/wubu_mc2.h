/* GROUP 2: Advanced Motion Compensation */
#ifndef WUBU_MC2_H
#define WUBU_MC2_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_gen_quarterpel(const uint8_t* src,uint8_t* dst,int W,int H);
void wubu_obmc(const uint8_t* ref,int W,int H,
                 const int16_t* mv_field,int blocks_per_row,int bs,
                 uint8_t* output);
int  wubu_temporal_mv(const int16_t* coloc_mv_field,int coloc_blocks_per_row,
                       int bx,int by,int16_t* out_mv);
typedef struct {
    uint8_t** frames;
    int* pocs;
    int* is_longterm;
    int capacity;
    int count;
} RefPool;
RefPool* wubu_refpool_create(int max_refs);
void wubu_refpool_push(RefPool* rp,const uint8_t* frame,int poc,int longterm);
const uint8_t* wubu_refpool_get(RefPool* rp,int ref_idx,int* is_longterm);
void wubu_refpool_evict(RefPool* rp,int max_shortterm);
#ifdef __cplusplus
}
#endif
#endif
