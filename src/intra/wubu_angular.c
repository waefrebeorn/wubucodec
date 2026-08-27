/*
 * wubu_angular.c -- GROUP 9: Angular Intra Prediction (HEVC-style)
 *
 * 35 prediction modes: Planar(0), DC(1), Angular(2-34)
 * Each angular mode has a displacement angle that determines how
 * reference pixels from the top row and left column project into
 * the block interior.
 *
 * The displacement table maps mode → (dx,dy) direction:
 * modes 2-17: mostly horizontal (project onto left column)
 * modes 18-33: mostly vertical (project onto top row)
 */
#include "wubu_angular.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Displacement table for angular modes 2-34 (from HEVC spec Table 8-2)
 * For vertical modes (18-34): displacement = horizontal offset per row
 * For horizontal modes (2-17): displacement = vertical offset per column
 */
static const int8_t ang_disp[35]={
    0,0,                                    /* modes 0,1 unused */
    -32,-26,-21,-17,-13,-9,-5,-2,0,         /* modes 2-10 (horizontal+) */
     2, 5, 9,13,17,21,26,32,                /* modes 11-17 (horizontal-) */
     0,                                      /* mode 18 = pure diagonal */
    -32,-26,-21,-17,-13,-9,-5,-2,0,         /* modes 19-27 (vertical) */
     2, 5, 9,13,17,21,26,32                  /* modes 28-34 (vertical+) */
};

/* Get reference samples: top row + left column + corner.
 * If unavailable, substitute with nearest available value. */
static void get_refs(const uint8_t* img,int W,int H,
                      int bx,int by,int bs,uint8_t* refs){
    /* refs[0] = corner (top-left), refs[1..bs] = top row, refs[bs+1..2*bs] = left col */
    int px=bx*bs,py=by*bs;
    
    /* corner */
    if(px>0&&py>0)refs[0]=img[(size_t)(py-1)*W+(px-1)];
    else if(px>0)refs[0]=img[(size_t)py*W+(px-1)];
    else if(py>0)refs[0]=img[(size_t)(py-1)*W+px];
    else refs[0]=128;
    
    /* top row */
    for(int c=0;c<bs;c++){
        if(py>0){
            int x=px+c;
            refs[c+1]=img[(size_t)(py-1)*W+(x<W?x:W-1)];
        }else{
            refs[c+1]=refs[0]; /* substitute with corner */
        }
    }
    
    /* left column */
    for(int r=0;r<bs;r++){
        if(px>0){
            int y=py+r;
            refs[bs+1+r]=img[(size_t)(y<H?y:H-1)*W+(px-1)];
        }else{
            refs[bs+1+r]=refs[0];
        }
    }
}

/* Planar prediction: bilinear surface fit using all border pixels */
static void pred_planar(const uint8_t* refs,uint8_t* block,int bs){
    uint8_t tl=refs[0],tr=refs[bs],bl=refs[bs+1],br=128;
    /* br is not directly available in our ref layout; approximate */
    br=(tr+bl)/2; /* simple approximation */
    
    for(int y=0;y<bs;y++)
        for(int x=0;x<bs;x++){
            int val=((bs-1-x)*tl+(x+1)*tr+
                     (bs-1-y)*tl+(y+1)*bl+
                     (x*y+((bs-1-x)*(bs-1-y)))*br)/(2*bs);
            if(val<0)val=0;if(val>255)val=255;
            block[y*bs+x]=(uint8_t)val;
        }
}

/* DC prediction: average of available border pixels */
static void pred_dc(const uint8_t* refs,uint8_t* block,int bs){
    int sum=0,count=0;
    for(int i=1;i<=bs;i++){sum+=refs[i];count++;}      /* top */
    for(int i=0;i<bs;i++){sum+=refs[bs+1+i];count++;}   /* left */
    uint8_t dc=(uint8_t)((sum+count/2)/count);
    for(int i=0;i<bs*bs;i++)block[i]=dc;
}

/* Angular prediction: project along the mode's direction */
static void pred_angular(const uint8_t* refs,uint8_t* block,
                          int bs,int mode){
    int disp=ang_disp[mode];
    
    if(mode>=18){
        /* vertical-ish modes: project each pixel onto the top reference row */
        for(int x=0;x<bs;x++){
            int offset=((x-(bs/2))*disp)>>5;
            int ref_idx=x+offset;
            if(ref_idx<0)ref_idx=0;
            if(ref_idx>bs-1)ref_idx=bs-1;
            uint8_t val=refs[ref_idx+1];
            for(int y=0;y<bs;y++)block[y*bs+x]=val;
        }
    }else{
        /* horizontal-ish modes: project onto the left reference column */
        for(int y=0;y<bs;y++){
            int offset=((y-(bs/2))*disp)>>5;
            int ref_idx=y+offset;
            if(ref_idx<0)ref_idx=0;
            if(ref_idx>bs-1)ref_idx=bs-1;
            uint8_t val=refs[bs+1+ref_idx];
            for(int x=0;x<bs;x++)block[y*bs+x]=val;
        }
    }
}

/* main intra prediction entry point */
void wubu_ipred(const uint8_t* img,int W,int H,
                 int bx,int by,int bs,
                 int mode,uint8_t* block){
    uint8_t refs[2*BS_MAX+1];
    get_refs(img,W,H,bx,by,bs,refs);
    
    if(mode==0)pred_planar(refs,block,bs);
    else if(mode==1)pred_dc(refs,block,bs);
    else pred_angular(refs,block,bs,mode);
}

/* select best intra mode by minimum SAD against actual pixels */
int wubu_ipred_best_mode(const uint8_t* img,int W,int H,
                           int bx,int by,int bs,
                           const uint8_t* actual){
    uint8_t pred[BS_MAX*BS_MAX];
    long best_sad=~(long)0;
    int best_mode=1; /* default to DC */
    
    for(int mode=0;mode<35;mode++){
        wubu_ipred(img,W,H,bx,by,bs,mode,pred);
        long sad=0;
        for(int i=0;i<bs*bs;i++)
            sad+=abs(actual[i]-(int)pred[i]);
        if(sad<best_sad){best_sad=sad;best_mode=mode;}
    }
    return best_mode;
}
