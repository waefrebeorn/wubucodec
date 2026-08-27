/*
 * wubu_deblock.c -- GAP-C073: Deblocking filter for DCT-based codec
 *
 * After quantized DCT reconstruction, block edges show discontinuities
 * ("blocking artifacts"). This adaptive filter smooths across boundaries
 * when the difference is small (artifact) but preserves real edges.
 *
 * Supports both RGB (3-channel) and single-plane (Y-only) modes.
 */
#include "wubu_deblock.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* apply deblocking to single-plane (Y-only) image */
static void db_filter_plane(uint8_t* img,int W,int H,int quality){
    /* threshold scales with quality (higher quality = gentler filtering) */
    int alpha=(quality<3)?24:(quality<6)?16:(quality<9)?8:4;
    int beta=alpha/2;

    /* vertical boundaries (between columns x=7,15,23,...) */
    for(int by=0;by<H;by+=8){
        for(int bx=8;bx<W;bx+=8){
            for(int r=0;r<8;r++){
                int y=by+r;
                if(y>=H)break;
                /* pixels on either side of boundary */
                int p2=img[y*W+bx-2];
                int p1=img[y*W+bx-1];
                int q1=img[y*W+bx];
                int q2=img[y*W+bx+1<W?bx+1:W-1];

                int d=p1-q1;
                if(abs(d)>=alpha)continue; /* real edge, don't touch */

                /* smoothness check on both sides */
                if(abs(p2-p1)<beta&&abs(q2-q1)<beta){
                    /* blend */
                    int delta=(d+2)>>2;
                    int v1=p1-delta,v2=q1+delta;
                    img[y*W+bx-1]=(uint8_t)(v1<0?0:(v1>255?255:v1));
                    if(bx<W)img[y*W+bx]=(uint8_t)(v2<0?0:(v2>255?255:v2));
                }
            }
        }
    }

    /* horizontal boundaries (between rows y=7,15,23,...) */
    for(int bx=0;bx<W;bx+=8){
        for(int by=8;by<H;by+=8){
            for(int c=0;c<8;c++){
                int x=bx+c;
                if(x>=W)break;
                int p2=(by-2>=0)?img[(by-2)*W+x]:img[x];
                int p1=img[(by-1)*W+x];
                int q1=img[by*W+x];
                int q2=(by+1<H)?img[(by+1)*W+x]:img[(H-1)*W+x];

                int d=p1-q1;
                if(abs(d)>=alpha)continue; /* real edge, don't touch */

                /* smoothness check on both sides */
                if(abs(p2-p1)<beta&&abs(q2-q1)<beta){
                    /* blend */
                    int delta=(d+2)>>2;
                    int v1=p1-delta,v2=q1+delta;
                    img[(by-1)*W+x]=(uint8_t)(v1<0?0:(v1>255?255:v1));
                    if(by<H)img[by*W+x]=(uint8_t)(v2<0?0:(v2>255?255:v2));
                }
            }
        }
    }
}

/* apply deblocking to RGB image (3 channels) */
void wubu_db_filter(uint8_t* img,int W,int H,int quality){
    /* Process each channel independently */
    for(int c=0;c<3;c++){
        /* Extract channel to temp plane */
        uint8_t* plane=malloc((size_t)W*H);
        if(!plane) return;
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                plane[y*W+x]=img[(y*W+x)*3+c];
        
        /* Filter plane */
        db_filter_plane(plane,W,H,quality);
        
        /* Write back */
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                img[(y*W+x)*3+c]=plane[y*W+x];
        
        free(plane);
    }
}

/* apply deblocking to single-plane image (Y-only) */
void wubu_db_filter_plane(uint8_t* img,int W,int H,int quality){
    db_filter_plane(img,W,H,quality);
}
