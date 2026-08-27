/*
 * wubu_intra.c -- GAP-C074: Intra prediction modes for KEY frames
 * (DC + vertical + horizontal + plane — the H.264 approach)
 *
 * KEY frames have no temporal reference. Instead of coding raw pixels,
 * predict each block from already-decoded neighboring blocks, then code
 * only the residual. This typically gives 30-50% better KEY compression.
 *
 * Four modes:
 *   0: DC    — average of available left+top pixels
 *   1: Vert  — copy row of pixels from above
 *   2: Horiz — copy column of pixels from left
 *   3: Plane — planar extrapolation (smooth gradient fit)
 */
#include "wubu_intra.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BS 8  /* block size */

/* check availability of neighbors */
static int has_left(const uint8_t* img,int W,int H,int bx,int by){
    return bx>0;
}
static int has_top(const uint8_t* img,int W,int H,int bx,int by){
    return by>0;
}

/* mode 0: DC — average of available border pixels */
static void intra_dc(uint8_t* img,int W,int H,int bx,int by,uint8_t* block){
    int sum=0,count=0;
    /* left column */
    if(bx>0){
        for(int r=0;r<BS;r++){
            int y=by*BS+r;if(y>=H)break;
            sum+=img[((size_t)y*W+(bx-1)*BS+7)*3];
            count++;
        }
    }
    /* top row */
    if(by>0){
        for(int c=0;c<BS;c++){
            int x=bx*BS+c;if(x>=W)break;
            sum+=img[((size_t)(by-1)*BS+7)*3+x*3];
            count++;
        }
    }
    uint8_t dc=(uint8_t)(count>0?sum/count:128);
    for(int i=0;i<64;i++)block[i]=dc;
}

/* mode 1: vertical — replicate top row downward */
static void intra_vert(const uint8_t* img,int W,int H,int bx,int by,uint8_t* block){
    if(by==0){intra_dc(img,W,H,bx,by,block);return;}
    for(int c=0;c<BS;c++){
        int x=bx*BS+c;if(x>=W)break;
        uint8_t v=img[((size_t)(by-1)*BS+7)*W*3+x*3];
        for(int r=0;r<BS;r++)block[r*BS+c]=v;
    }
}

/* mode 2: horizontal — replicate left column rightward */
static void intra_horiz(const uint8_t* img,int W,int H,int bx,int by,uint8_t* block){
    if(bx==0){intra_dc(img,W,H,bx,by,block);return;}
    for(int r=0;r<BS;r++){
        int y=by*BS+r;if(y>=H)break;
        uint8_t v=img[((size_t)y*W+(bx-1)*BS+7)*3];
        for(int c=0;c<BS;c++)block[r*BS+c]=v;
    }
}

/* mode 3: plane — weighted extrapolation from corner */
static void intra_plane(uint8_t* img,int W,int H,int bx,int by,uint8_t* block){
    if(bx==0||by==0){
        intra_vert(img,W,H,bx,by,block);
        return;
    }
    /* use corner pixel and gradient estimate */
    int corner=img[((size_t)((by-1)*BS+7)*W+(bx-1)*BS+7)*3];
    int grad_h=0,grad_v=0,n=0;
    for(int i=0;i<BS&&(bx-1)*BS+7+i<W;i++){
        int v=img[((size_t)((by-1)*BS+7)*W+(bx-1)*BS+7+i)*3];
        grad_h+=v-corner;n++;
    }
    for(int i=0;i<BS&&(by-1)*BS+7+i<H;i++){
        int v=img[((size_t)((by-1)*BS+7+i)*W+(bx-1)*BS+7)*3];
        grad_v+=v-corner;n++;
    }
    if(n>0){grad_h/=n;grad_v/=n;}

    for(int r=0;r<BS;r++)
        for(int c=0;c<BS;c++){
            int v=corner+(grad_h*(c+1))/BS+(grad_v*(r+1))/BS;
            block[r*BS+c]=(uint8_t)(v<0?0:(v>255?255:v));
        }
}

/* predict a single 8x8 block with the specified mode */
void wubu_ip_predict(uint8_t* img,int W,int H,int bx,int by,
                      int mode,uint8_t* block){
    switch(mode){
        case 0: intra_dc(img,W,H,bx,by,block);break;
        case 1: intra_vert(img,W,H,bx,by,block);break;
        case 2: intra_horiz(img,W,H,bx,by,block);break;
        case 3: intra_plane(img,W,H,bx,by,block);break;
        default: intra_dc(img,W,H,bx,by,block);
    }
}

/* auto-select best mode by minimizing SAD against actual block */
int wubu_ip_best_mode(const uint8_t* img,const uint8_t* actual,
                       int W,int H,int bx,int by){
    uint8_t pred_block[64];
    int best_mode=0;
    long best_sad=1LL<<40;

    for(int mode=0;mode<4;mode++){
        wubu_ip_predict((uint8_t*)img,W,H,bx,by,mode,pred_block);
        long sad=0;
        for(int i=0;i<64;i++)
            sad+=abs(actual[i]-(int)pred_block[i]);
        if(sad<best_sad){best_sad=sad;best_mode=mode;}
    }
    return best_mode;
}
