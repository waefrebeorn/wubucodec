/* wubu_satd.c — SATD (Sum of Absolute Transformed Differences) for mode decision */
#include "wubu_satd.h"
#include <stdlib.h>
#include <string.h>

/* 4x4 Hadamard transform (2D) */
static void hadamard_4x4(int16_t* blk){
    /* Row transform */
    for(int r = 0; r < 4; r++){
        int16_t* row = blk + r*4;
        int a0 = row[0], a1 = row[1], a2 = row[2], a3 = row[3];
        row[0] = a0 + a1 + a2 + a3;
        row[1] = a0 + a1 - a2 - a3;
        row[2] = a0 - a1 - a2 + a3;
        row[3] = a0 - a1 + a2 - a3;
    }
    /* Column transform */
    for(int c = 0; c < 4; c++){
        int16_t* col = blk + c;
        int a0 = col[0], a1 = col[4], a2 = col[8], a3 = col[12];
        col[0]  = a0 + a1 + a2 + a3;
        col[4]  = a0 + a1 - a2 - a3;
        col[8]  = a0 - a1 - a2 + a3;
        col[12] = a0 - a1 + a2 - a3;
    }
}

/* SATD for 4x4 block: sum of absolute Hadamard-transformed differences */
int wubu_satd_4x4(const uint8_t* a, const uint8_t* b, int stride_a, int stride_b){
    int16_t diff[16];
    int satd = 0;
    
    /* Compute residual */
    for(int r = 0; r < 4; r++)
        for(int c = 0; c < 4; c++)
            diff[r*4+c] = (int16_t)((int)a[r*stride_a+c] - (int)b[r*stride_b+c]);
    
    /* Hadamard transform */
    hadamard_4x4(diff);
    
    /* Sum of absolute values */
    for(int i = 0; i < 16; i++)
        satd += abs(diff[i]);
    
    return satd;
}

/* SATD for 8x8 block: split into four 4x4 sub-blocks */
int wubu_satd_8x8(const uint8_t* a, const uint8_t* b, int stride_a, int stride_b){
    int satd = 0;
    for(int sy = 0; sy < 8; sy += 4)
        for(int sx = 0; sx < 8; sx += 4)
            satd += wubu_satd_4x4(a + sy*stride_a + sx, b + sy*stride_b + sx, stride_a, stride_b);
    return satd;
}

/* Fast SATD approximation: skip Hadamard, use SAD of high-pass filtered residual */
int wubu_satd_fast_8x8(const uint8_t* a, const uint8_t* b, int stride_a, int stride_b){
    int satd = 0;
    for(int r = 0; r < 8; r++){
        for(int c = 0; c < 8; c++){
            int diff = (int)a[r*stride_a+c] - (int)b[r*stride_b+c];
            /* High-pass weighting: emphasize edges */
            if(r > 0) diff -= (int)a[(r-1)*stride_a+c] - (int)b[(r-1)*stride_b+c];
            if(c > 0) diff -= (int)a[r*stride_a+c-1] - (int)b[r*stride_b+c-1];
            satd += abs(diff);
        }
    }
    return satd;
}
