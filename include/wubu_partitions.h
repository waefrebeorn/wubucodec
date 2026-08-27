/* GROUP 1: Partitions, search patterns, early termination */
#ifndef WUBU_PARTITIONS_H
#define WUBU_PARTITIONS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PART_16x16,PART_16x8,PART_8x16,PART_8x8,PART_8x4,PART_4x8,PART_4x4,PART_AMP
} wubu_part_type;

typedef struct {
    int x,y,w,h;
    wubu_part_type type;
} wubu_partition_t;

long wubu_block_variance(const uint8_t* img,int W,int H,
                          int bx,int by,int bs);
int  wubu_enum_partitions(int block_size,wubu_partition_t* parts);
long wubu_diamond_search(const uint8_t* curr,const uint8_t* ref,
                           int W,int H,int bx,int by,int bs,
                           int search_range,int* out_dx,int* out_dy);
long wubu_hexagon_search(const uint8_t* curr,const uint8_t* ref,
                           int W,int H,int bx,int by,int bs,
                           int search_range,int* out_dx,int* out_dy);
#ifdef __cplusplus
}
#endif
#endif
