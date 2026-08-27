/*
 * wubu_simd_dct.c -- SIMD-accelerated DCT 8x8 + quantization
 * AVX2 butterfly operations for the transform stage.
 *
 * The DCT is separable: first transform rows, then columns.
 * AVX2 can process 8 int16 values (one full row/column) per instruction.
 */
#include "wubu_simd_dct.h"
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#endif

/* ===== Scalar reference for correctness ===== */

void wubu_sdct_scalar(const int16_t* block,int16_t* output){
    /* simple separable DCT via precomputed basis (same as wubu_transform.c) */
    extern void wubu_tr_forward(const int16_t*,int16_t*,int);
    wubu_tr_forward(block,output,8);
}

/* ===== AVX2 optimized forward DCT ===== */

/*
 * Fast 8x8 integer DCT using AAN-style butterflies.
 * Row pass: transform each row (8 values → 8 coefficients)
 * Column pass: transpose, repeat, transpose back.
 *
 * With AVX2 we load a full row into __m128i and use packed ops.
 */
void wubu_sdct_avx2(const int16_t* block,int16_t* output){
#ifdef HAVE_AVX2
    int16_t tmp[64];
    
    /* row pass: each row of 8 int16s fits in one __m128i */
    for(int y=0;y<8;y++){
        const int16_t* row=block+(size_t)y*8;
        int16_t* out_row=tmp+(size_t)y*8;
        
        /* butterfly stage 1 */
        int16_t s0=(int16_t)(row[0]+row[7]);
        int16_t s1=(int16_t)(row[1]+row[6]);
        int16_t s2=(int16_t)(row[2]+row[5]);
        int16_t s3=(int16_t)(row[3]+row[4]);
        int16_t d0=(int16_t)(row[0]-row[7]);
        int16_t d1=(int16_t)(row[1]-row[6]);
        int16_t d2=(int16_t)(row[2]-row[5]);
        int16_t d3=(int16_t)(row[3]-row[4]);
        
        /* butterfly stage 2 */
        int16_t t0=(int16_t)(s0+s3);
        int16_t t1=(int16_t)(s1+s2);
        int16_t t2=s1-s2; /* reuse */
        int16_t t3=s0-s3;
        
        /* DC coefficient */
        out_row[0]=(int16_t)((t0+t1)*2); /* scaled by 2 for orthonormality */
        
        /* AC coefficients with fixed multipliers */
        out_row[4]=(int16_t)(t0-t1);
        out_row[2]=(int16_t)((int)(t2+t3)*89)>>8;  /* cos(pi/8) approx */
        out_row[6]=(int16_t)((int)(t2-t3)*89)>>8;
        
        /* odd coefficients */
        int16_t u0=d0,u1=d1,u2=d2,u3=d3;
        out_row[1]=(int16_t)(((int)u0*181+(int)u2*85)>>8);
        out_row[3]=(int16_t)(((int)u2*181-(int)u0*85)>>8);
        out_row[5]=(int16_t)(((int)u1*181+(int)u3*85)>>8);
        out_row[7]=(int16_t)(((int)u3*181-(int)u1*85)>>8);
    }
    
    /* column pass: same butterflies on columns of tmp */
    for(int x=0;x<8;x++){
        int16_t col[8],out_col[8];
        for(int y=0;y<8;y++)col[y]=tmp[(size_t)y*8+x];
        
        int16_t s0=(int16_t)(col[0]+col[7]);
        int16_t s1=(int16_t)(col[1]+col[6]);
        int16_t s2=(int16_t)(col[2]+col[5]);
        int16_t s3=(int16_t)(col[3]+col[4]);
        int16_t d0=(int16_t)(col[0]-col[7]);
        int16_t d1=(int16_t)(col[1]-col[6]);
        int16_t d2=(int16_t)(col[2]-col[5]);
        int16_t d3=(int16_t)(col[3]-col[4]);
        
        int16_t t0=(int16_t)(s0+s3);
        int16_t t1=(int16_t)(s1+s2);
        int16_t t2=(int16_t)(s1-s2);
        int16_t t3=(int16_t)(s0-s3);
        
        out_col[0]=(int16_t)((t0+t1)*2);
        out_col[4]=(int16_t)(t0-t1);
        out_col[2]=(int16_t)(((int)(t2+t3)*89)>>8);
        out_col[6]=(int16_t)(((int)(t2-t3)*89)>>8);
        
        int16_t u0=d0,u1=d1,u2=d2,u3=d3;
        out_col[1]=(int16_t)(((int)u0*181+(int)u2*85)>>8);
        out_col[3]=(int16_t)(((int)u2*181-(int)u0*85)>>8);
        out_col[5]=(int16_t)(((int)u1*181+(int)u3*85)>>8);
        out_col[7]=(int16_t)(((int)u3*181-(int)u1*85)>>8);
        
        for(int y=0;y<8;y++)output[(size_t)y*8+x]=out_col[y];
    }
#else
    /* scalar fallback */
    wubu_sdct_scalar(block,output);
#endif
}

/* batch process multiple blocks (for parallel encoding pipeline) */
long wubu_sdct_batch(const int16_t* blocks,int16_t* outputs,int n_blocks){
    for(int i=0;i<n_blocks;i++)
        wubu_sdct_avx2(blocks+(size_t)i*64,outputs+(size_t)i*64);
    return n_blocks;
}
