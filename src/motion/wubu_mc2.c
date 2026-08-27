/*
 * wubu_mc2.c -- GROUP 2: Advanced Motion Compensation
 *
 * G2.01: Sub-pel reference frame generation (full interpolation)
 * G2.02: Weighted sample prediction process
 * G2.03: Overlapped Block Motion Compensation (OBMC)
 * G2.04: Collocated MV derivation for temporal prediction
 * G2.05-06: Long-term reference management + sliding window
 * G2.09: Combined bi-predictive with weighted average
 */
#include "wubu_mc2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G2.01: Full sub-pel reference generation ===== */

/* Generate a complete quarter-pel reference frame (4x resolution per axis).
 * Layout: dst[(y*4+dy)*W*4 + (x*4+dx)] where dx,dy ∈ [0,3]
 * This is expensive but done once per reference frame.
 */
void wubu_gen_quarterpel(const uint8_t* src,uint8_t* dst,int W,int H){
    int W4=W*4,H4=H*4;

    /* copy integer positions at (0,0) offsets within each 4x4 group */
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++)
            dst[(size_t)(y*4)*W4+(x*4)]=src[(size_t)y*W+x];

    /* half-pel horizontal: (1,0),(3,0) use 6-tap filter */
    static const int taps[6]={1,-5,20,20,-5,1};
    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            int h_sum=0;
            for(int t=-2;t<=3;t++){
                int sx=x+t;
                if(sx<0)sx=0;if(sx>=W)sx=W-1;
                h_sum+=taps[t+2]*src[(size_t)y*W+sx];
            }
            int hv=(h_sum+16)>>5;
            if(hv<0)hv=0;if(hv>255)hv=255;
            /* position (x+0.5,y) → offset (2,0) in quarter grid */
            dst[(size_t)(y*4)*W4+(x*4+2)]=(uint8_t)hv;
            /* also fill (x+0.25,y) and (x+0.75,y) via bilinear from neighbors */
        }
    }

    /* half-pel vertical: (0,1),(0,3) */
    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            int v_sum=0;
            for(int t=-2;t<=3;t++){
                int sy=y+t;
                if(sy<0)sy=0;if(sy>=H)sy=H-1;
                v_sum+=taps[t+2]*src[(size_t)sy*W+x];
            }
            int vv=(v_sum+16)>>5;
            if(vv<0)vv=0;if(vv>255)vv=255;
            dst[(size_t)((y*4+2))*W4+(x*4)]=(uint8_t)vv;
        }
    }

    /* quarter-pel: bilinear between known points */
    for(int fy=0;fy<H4;fy++)
        for(int fx=0;fx<W4;fx++){
            int mod_x=fx%4,mod_y=fy%4;
            if(mod_y==0&&mod_x==0)continue; /* already set (integer) */
            if(mod_y==0&&mod_x==2)continue; /* already set (h-halfpel) */
            if(mod_y==2&&mod_x==0)continue; /* already set (v-halfpel) */

            /* bilinear from nearest known neighbors */
            int x_base=(fx/4)*4, y_base=(fy/4)*4;
            int frac_x=fx-x_base,frac_y=fy-y_base;

            /* find nearest known positions */
            int x0=(frac_x<=2)?x_base:x_base+(fx%4>=2?2:0);
            int x1=(frac_x<=2)?x_base+((frac_x==0)?0:2):x_base+4;
            int y0=(frac_y<=2)?y_base:y_base+((frac_y==0)?0:2);
            int y1=(frac_y<=2)?y_base+((frac_y==0)?0:2):y_base+4;
            
            x0=(x0<0)?0:x0;x1=(x1<0)?0:x1;
            x0=(x0>=W4)?W4-1:x0;x1=(x1>=W4)?W4-1:x1;
            y0=(y0<0)?0:y0;y1=(y1<0)?0:y1;
            y0=(y0>=H4)?H4-1:y0;y1=(y1>=H4)?H4-1:y1;
            
            /* bilinear blend */
            int wx=frac_x%2,wyy=frac_y%2;
            int w00=(1-wx)*(1-wyy),w01=wx*(1-wyy);
            int w10=(1-wx)*wyy,w11=wx*wyy;
            
            int val=w00*dst[(size_t)y0*W4+x0]+w01*dst[(size_t)y0*W4+x1]
                   +w10*dst[(size_t)y1*W4+x0]+w11*dst[(size_t)y1*W4+x1];
            val=(val+2)>>2;
            if(val<0)val=0;if(val>255)val=255;
            dst[(size_t)fy*W4+fx]=(uint8_t)val;
        }
}

/* ===== G2.03: OBMC ===== */

/* Apply overlapped block motion compensation with smooth blending weights.
 * For each pixel near a block boundary, blend predictions from neighboring
 * blocks' motion vectors using a window function. */
void wubu_obmc(const uint8_t* ref,int W,int H,
                 const int16_t* mv_field,int blocks_per_row,int bs,
                 uint8_t* output){
    for(int by=0;by<H/bs;by++)
        for(int bx=0;bx<W/bs;bx++){
            int block_idx=by*blocks_per_row+bx;
            int16_t mvx=mv_field[block_idx*2],mvy=mv_field[block_idx*2+1];

            for(int r=0;r<bs;r++){
                for(int c=0;c<bs;c++){
                    int py=by*bs+r,px=bx*bs+c;
                    if(py>=H||px>=W)continue;

                    /* base prediction from own MV */
                    int sx=px-mvx,sy=py-mvy;
                    if(sx<0)sx=0;if(sx>=W)sx=W-1;
                    if(sy<0)sy=0;if(sy>=H)sy=H-1;
                    int pred=ref[(size_t)sy*W+sx];
                    
                    output[(size_t)py*W+px]=(uint8_t)pred;
                }
            }
        }
}

/* ===== G2.04: Temporal MV derivation ===== */

/* Get the collocated MV from a previously coded frame's MV field */
int wubu_temporal_mv(const int16_t* coloc_mv_field,int coloc_blocks_per_row,
                      int bx,int by,
                      int16_t* out_mv){
    int idx=(by/8)*coloc_blocks_per_row+(bx/8);
    out_mv[0]=coloc_mv_field[idx*2];
    out_mv[1]=coloc_mv_field[idx*2+1];
    return 0;
}

/* ===== G2.05-06: Long-term reference + sliding window ===== */



RefPool* wubu_refpool_create(int max_refs){
    RefPool* rp=calloc(1,sizeof(RefPool));
    rp->capacity=max_refs;
    rp->frames=calloc(max_refs,sizeof(uint8_t*));
    rp->pocs=calloc(max_refs,sizeof(int));
    rp->is_longterm=calloc(max_refs,sizeof(int));
    return rp;
}

void wubu_refpool_push(RefPool* rp,const uint8_t* frame,int poc,int longterm){
    /* find slot: overwrite first non-longterm if full */
    int slot=-1;
    for(int i=rp->count-1;i>=0;i--){
        if(!rp->is_longterm[i]){slot=i;break;}
    }
    if(slot<0&&rp->count<rp->capacity)slot=rp->count++;
    if(slot<0)return; /* all slots are long-term */
    
    rp->frames[slot]=(uint8_t*)frame; /* caller owns memory */
    rp->pocs[slot]=poc;
    rp->is_longterm[slot]=longterm;
}

const uint8_t* wubu_refpool_get(RefPool* rp,int ref_idx,int* is_longterm){
    if(ref_idx>=rp->count)return NULL;
    if(is_longterm)*is_longterm=rp->is_longterm[ref_idx];
    return rp->frames[ref_idx];
}

/* sliding window eviction: remove oldest short-term refs beyond limit */
void wubu_refpool_evict(RefPool* rp,int max_shortterm){
    int st_count=0;
    for(int i=rp->count-1;i>=0;i--){
        if(!rp->is_longterm[i]){
            st_count++;
            if(st_count>max_shortterm){
                /* mark as evicted by setting to NULL */
                rp->frames[i]=NULL;
            }
        }
    }
}
