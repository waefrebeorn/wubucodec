/*
 * wubu_motionest.c -- GAP-C075: Block matching motion estimation
 * with quarter-pel refinement.
 *
 * Pipeline:
 *   1. Integer-pel diamond search (fast, near-optimal)
 *   2. Quarter-pel refinement around best integer MV
 *   3. Bilinear interpolation for sub-pel positions
 *
 * Quarter-pel positions around best integer MV (dx,dy):
 *   (dx,dy), (dx+0.25,dy), (dx+0.5,dy), (dx+0.75,dy)
 *   (dx,dy+0.25), ..., (dx+0.75,dy+0.75)
 * This gives 16 positions in a 4x4 grid around the integer MV.
 */
#include "wubu_motionest.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* SAD between a block in curr at (bx,by) and ref at (bx+dx,by+dy) */
long wubu_me_sad(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,int dx,int dy){
    long sad=0;
    for(int r=0;r<bs;r++){
        int cy=by+r;
        int ry=cy+dy;
        if(ry<0||ry>=H){return 1LL<<30;}
        for(int c=0;c<bs;c++){
            int cx=bx+c;
            int rx=cx+dx;
            if(rx<0||rx>=W){sad+=(1<<20);continue;}
            sad+=abs(curr[cy*W+cx]-ref[ry*W+rx]);
        }
    }
    return sad;
}

/* Bilinearly interpolated SAD for quarter-pel positions.
 * dx,dy are in quarter-pel units (e.g., dx=1 means 0.25 pel, dx=2 means 0.5 pel). */
static long wubu_me_sad_qpel(const uint8_t* curr,const uint8_t* ref,
                               int W,int H,int bx,int by,int bs,
                               int dx_qpel,int dy_qpel){
    /* dx_qpel, dy_qpel in quarter-pel units. Actual offset = dx_qpel/4.0, dy_qpel/4.0 */
    long sad=0;
    for(int r=0;r<bs;r++){
        for(int c=0;c<bs;c++){
            int cx=bx+c;
            int cy=by+r;
            /* Sub-pel position in the reference */
            float rx_f = (float)cx + (float)dx_qpel * 0.25f;
            float ry_f = (float)cy + (float)dy_qpel * 0.25f;
            int rx0 = (int)floorf(rx_f);
            int ry0 = (int)floorf(ry_f);
            float fx = rx_f - (float)rx0;
            float fy = ry_f - (float)ry0;
            /* Bilinear interpolation */
            if(ry0<0||ry0>=H||rx0<0||rx0>=W){sad+=(1<<20);continue;}
            int v00 = ref[((size_t)ry0*W+rx0)*3];
            int v10 = (rx0+1<W) ? ref[((size_t)ry0*W+rx0+1)*3] : v00;
            int v01 = (ry0+1<H) ? ref[((size_t)(ry0+1)*W+rx0)*3] : v00;
            int v11 = (rx0+1<W&&ry0+1<H) ? ref[((size_t)(ry0+1)*W+rx0+1)*3] : v00;
            float v = (float)v00*(1.0f-fx)*(1.0f-fy) +
                      (float)v10*fx*(1.0f-fy) +
                      (float)v01*(1.0f-fx)*fy +
                      (float)v11*fx*fy;
            int v_int = (int)(v+0.5f);
            if(v_int<0)v_int=0; if(v_int>255)v_int=255;
            sad+=abs(curr[((size_t)cy*W+cx)*3]-v_int);
        }
    }
    return sad;
}

/* find best motion vector for one block using diamond + quarter-pel refinement */
long wubu_me_block(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,
                    int search_range,int* out_dx,int* out_dy){
    /* Phase 1: Integer-pel diamond search */
    int best_dx=0,best_dy=0;
    long best_sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,0,0);

    static const int dia_dx[]={0,-1,0,1,0};
    static const int dia_dy[]={-1,0,1,0,-1};

    int improved=1;
    while(improved){
        improved=0;
        for(int d=0;d<4;d++){
            int ndx=best_dx+dia_dx[d];
            int ndy=best_dy+dia_dy[d];
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){
                best_sad=sad;best_dx=ndx;best_dy=ndy;
                improved=1;
            }
        }
    }

    /* Phase 2: Quarter-pel refinement around best integer MV */
    /* Search in quarter-pel units: -1, 0, 1, 2, 3 (in QPEL units) */
    /* 0 = integer, 2 = half-pel, 1 = quarter-pel, 3 = three-quarter-pel */
    int best_hx=0, best_hy=0;
    long best_sad_qpel=best_sad;
    for(int dy=-1;dy<=3;dy++){
        for(int dx=-1;dx<=3;dx++){
            if(dx==0&&dy==0)continue;
            long sad=wubu_me_sad_qpel(curr,ref,W,H,bx,by,bs,dx,dy);
            if(sad<best_sad_qpel){
                best_sad_qpel=sad;
                best_hx=dx;
                best_hy=dy;
            }
        }
    }

    /* Combine: integer + quarter-pel (in quarter-pel units) */
    *out_dx = best_dx*4 + best_hx;
    *out_dy = best_dy*4 + best_hy;
    return best_sad_qpel;
}

/* estimate motion vectors for all blocks in a frame */
int wubu_me_frame(const uint8_t* curr,const uint8_t* ref,
                   int W,int H,int bs,int search_range,
                   int* out_mvs /* [n_blocks_x*n_blocks_y][2], in quarter-pel units */){
    int nbx=W/bs,nby=H/bs;
    int count=0;
    for(int by=0;by<nby;by++)
        for(int bx=0;bx<nbx;bx++){
            int dx,dy;
            wubu_me_block(curr,ref,W,H,bx*bs,by*bs,bs,search_range,&dx,&dy);
            out_mvs[count*2]=dx;
            out_mvs[count*2+1]=dy;
            count++;
        }
    return count;
}

/* apply motion compensation: build predicted frame from reference + MVs.
 * MVs are in quarter-pel units (divide by 4.0 to get actual offset). */
void wubu_me_compensate(const uint8_t* ref,int W,int H,int bs,
                          const int* mvs,uint8_t* predicted){
    int nbx=W/bs,nby=H/bs;
    int count=0;
    for(int by=0;by<nby;by++)
        for(int bx=0;bx<nbx;bx++){
            int dx_qpel=mvs[count*2],dy_qpel=mvs[count*2+1];
            count++;
            for(int r=0;r<bs;r++){
                int sy=by*bs+r;
                for(int c=0;c<bs;c++){
                    int sx=bx*bs+c;
                    /* Sub-pel position in reference */
                    float rx_f = (float)sx + (float)dx_qpel * 0.25f;
                    float ry_f = (float)sy + (float)dy_qpel * 0.25f;
                    int rx0 = (int)floorf(rx_f);
                    int ry0 = (int)floorf(ry_f);
                    float fx = rx_f - (float)rx0;
                    float fy = ry_f - (float)ry0;
                    if(ry0<0||ry0>=H||rx0<0||rx0>=W){
                        for(int ch=0;ch<3;ch++)
                            predicted[((size_t)sy*W+sx)*3+ch]=0;
                        continue;
                    }
                    for(int ch=0;ch<3;ch++){
                        int v00 = ref[((size_t)ry0*W+rx0)*3+ch];
                        int v10 = (rx0+1<W) ? ref[((size_t)ry0*W+rx0+1)*3+ch] : v00;
                        int v01 = (ry0+1<H) ? ref[((size_t)(ry0+1)*W+rx0)*3+ch] : v00;
                        int v11 = (rx0+1<W&&ry0+1<H) ? ref[((size_t)(ry0+1)*W+rx0+1)*3+ch] : v00;
                        float v = (float)v00*(1.0f-fx)*(1.0f-fy) +
                                  (float)v10*fx*(1.0f-fy) +
                                  (float)v01*(1.0f-fx)*fy +
                                  (float)v11*fx*fy;
                        int v_int = (int)(v+0.5f);
                        if(v_int<0)v_int=0; if(v_int>255)v_int=255;
                        predicted[((size_t)sy*W+sx)*3+ch]=(uint8_t)v_int;
                    }
                }
            }
        }
}
