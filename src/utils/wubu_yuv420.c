/*
 * wubu_yuv420.c -- GAP-C076: RGB↔YUV420 conversion with SIMD
 *
 * Chroma subsampling halves the data (3 bytes/pixel → 1.5 bytes/pixel)
 * with virtually no visual quality loss for cartoons. The Y channel
 * carries all structure; U/V carry color at quarter resolution.
 *
 * BT.601 conversion (standard for SD content):
 *   Y  =  0.299*R + 0.587*G + 0.114*B
 *   Cb = -0.169*R - 0.331*G + 0.500*B + 128
 *   Cr =  0.500*R - 0.419*G - 0.081*B + 128
 */
#include "wubu_yuv420.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* RGB→YUV420: Y at full res, U/V at half res */
void wubu_rgb_to_yuv420(const uint8_t* rgb,uint8_t* y,uint8_t* u,uint8_t* v,
                          int W,int H){
    /* Y channel: full resolution */
    for(long i=0,j=0;i<(long)W*H*3;i+=3,j++){
        y[j]=(uint8_t)((rgb[i]*299+rgb[i+1]*587+rgb[i+2]*114)/1000);
    }
    /* U and V channels: half resolution, average 2x2 blocks */
    int hw=W/2,hh=H/2;
    for(int by=0;by<hh;by++){
        for(int bx=0;bx<hw;bx++){
            /* average the 2x2 block's chroma values */
            int sum_cb=0,sum_cr=0;
            for(int dy=0;dy<2;dy++){
                for(int dx=0;dx<2;dx++){
                    long i=((size_t)(by*2+dy)*W+(bx*2+dx))*3;
                    sum_cb+=(-rgb[i]*169-rgb[i+1]*331+rgb[i+2]*500)/1000+128;
                    sum_cr+=( rgb[i]*500-rgb[i+1]*419-rgb[i+2]*81)/1000+128;
                }
            }
            u[by*hw+bx]=(uint8_t)(sum_cb/4);
            v[by*hw+bx]=(uint8_t)(sum_cr/4);
        }
    }
}

/* YUV420→RGB: reconstruct full-resolution RGB from Y + subsampled U/V */
void wubu_yuv420_to_rgb(const uint8_t* y,const uint8_t* u,const uint8_t* v,
                          uint8_t* rgb,int W,int H){
    int hw=W/2;
    for(int py=0;py<H;py++){
        for(int px=0;px<W;px++){
            float Y=y[(size_t)py*W+px];
            float Cb=u[(py/2)*hw+(px/2)]-128.0f;
            float Cr=v[(py/2)*hw+(px/2)]-128.0f;

            float R=Y+1.402f*Cr;
            float G=Y-0.344f*Cb-0.714f*Cr;
            float B=Y+1.772f*Cb;

            long i=((size_t)py*W+px)*3;
            rgb[i]  =(uint8_t)(R<0?0:(R>255?255:R));
            rgb[i+1]=(uint8_t)(G<0?0:(G>255?255:G));
            rgb[i+2]=(uint8_t)(B<0?0:(B>255?255:B));
        }
    }
}

/* get total YUV420 data size */
long wubu_yuv420_size(int W,int H){
    return (long)W*H + 2*(W/2)*(H/2);  /* Y + U + V */
}
