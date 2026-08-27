/*
 * wubu_subpel.c -- GROUP 1: Sub-pixel motion estimation
 * Half-pel (6-tap Wiener) and quarter-pel (bilinear) interpolation.
 *
 * The H.264/AVC half-pel filter is the standard [1,-5,20,20,-5,1]/32
 * Wiener filter applied horizontally then vertically. Quarter-pel uses
 * bilinear from the half-pel grid. Eighth-pel is VVC-level precision.
 *
 * This module generates interpolated reference frames at half-pel and
 * quarter-pel positions, enabling sub-pixel motion vectors for ME.
 */
#include "wubu_subpel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* clip to [0,255] */
static uint8_t sp_clip(int x){
    return (uint8_t)(x<0?0:(x>255?255:x));
}

/*
 * Generate half-pel interpolated frame.
 * Layout: original pixels at even positions, interpolated at odd.
 * Output size = W*2 × H*2 (each dimension doubled).
 * Positions: (0,0)=orig, (1,0)=H-halfpel, (0,1)=V-halfpel, (1,1)=diag-halfpel
 */
void wubu_sp_halfpel(const uint8_t* src,uint8_t* dst,int W,int H){
    int W2=W*2,H2=H*2;

    /* horizontal 6-tap filter: H at position (2x+1, 2y) */
    static const int h_taps[6]={1,-5,20,20,-5,1};
    
    /* copy original pixels to even-even positions */
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++)
            dst[(size_t)(y*2)*W2+(x*2)]=src[(size_t)y*W+x];

    /* horizontal half-pel: (odd_x, even_y) */
    for(int y=0;y<H;y++){
        int dy=y*2;
        for(int x=0;x<W;x++){
            int dx=x*2+1;
            int sum=0;
            for(int t=-2;t<=3;t++){
                int sx=x+t;
                if(sx<0)sx=0;if(sx>=W)sx=W-1;
                sum+=h_taps[t+2]*src[(size_t)y*W+sx];
            }
            dst[(size_t)dy*W2+dx]=sp_clip((sum+16)>>5);
        }
    }

    /* vertical half-pel: (even_x, odd_y) — same filter vertically */
    for(int y=0;y<H;y++){
        int dy=y*2+1;
        for(int x=0;x<W;x++){
            int dx=x*2;
            int sum=0;
            for(int t=-2;t<=3;t++){
                int sy=y+t;
                if(sy<0)sy=0;if(sy>=H)sy=H-1;
                sum+=h_taps[t+2]*src[(size_t)sy*W+x];
            }
            dst[(size_t)dy*W2+dx]=sp_clip((sum+16)>>5);
        }
    }

    /* diagonal half-pel: (odd_x, odd_y) = vertical filter of horizontal results */
    for(int y=0;y<H;y++){
        int dy=y*2+1;
        for(int x=0;x<W;x++){
            int dx=x*2+1;
            int sum=0;
            for(int t=-2;t<=3;t++){
                int sy=y*2+1+(t*2); /* step by 2 in doubled space */
                if(sy<0)sy=0;if(sy>=H2)sy=H2-1;
                sum+=h_taps[t+2]*dst[(size_t)sy*W2+dx];
            }
            dst[(size_t)dy*W2+dx]=sp_clip((sum+16)>>5);
        }
    }
}

/*
 * Get pixel value at fractional position (quarter-pel precision).
 * dx,dy in quarter-pel units: 0..3 for each axis within a full pel.
 * Uses bilinear from nearest half-pel grid points.
 */
uint8_t wubu_sp_get(const uint8_t* hp,int W2,int H2,
                     int x_int,int y_int,int dx,int dy){
    /* convert to half-pel coordinates */
    int hx=x_int*2+(dx>=2?1:0);
    int hy=y_int*2+(dy>=2?1:0);

    if(dx==0&&dy==0)
        return hp[hy*W2+hx]; /* exact integer position */

    /* bilinear between neighboring half-pel samples */
    int x0=hx,x1=hx+((dx%2)?1:-1);
    int y0=hy,y1=hy+((dy%2)?1:-1);
    x0=(x0<0)?0:x0;x1=(x1<0)?0:x1;
    x0=(x0>=W2)?W2-1:x0;x1=(x1>=W2)?W2-1:x1;
    y0=(y0<0)?0:y0;y1=(y1<0)?0:y1;
    y0=(y0>=H2)?H2-1:y0;y1=(y1>=H2)?H2-1:y1;

    /* weights based on fraction */
    int wx=(dx%2)?1:0;  /* halfway or not */
    int wy=(dy%2)?1:0;
    int w00=(1-wx)*(1-wy), w01=wx*(1-wy);
    int w10=(1-wx)*wy,   w11=wx*wy;

    int val=w00*hp[y0*W2+x0]+w01*hp[y0*W2+x1]
           +w10*hp[y1*W2+x0]+w11*hp[y1*W2+x1];
    return sp_clip((val+2)>>2);
}

/*
 * Sub-pixel motion estimation: search over integer + half + quarter pel.
 * Returns best MV with quarter-pel precision.
 */
long wubu_sp_me(const uint8_t* curr,const uint8_t* hp_ref,
                  int W2,int H2,int bx,int by,int bs,
                  int search_range,
                  int* out_dx,int* out_dy){
    long best_sad=(long)1<<40;
    int best_dx=0,best_dy=0; /* in quarter-pel units */

    int W_orig=W2/2,H_orig=H2/2;
    /* integer-pel search first (coarse) */
    for(int dy=-search_range;dy<=search_range;dy++)
        for(int dx=-search_range;dx<=search_range;dx++){
            long sad=0;
            for(int r=0;r<bs;r++){
                int cy=by+r,ry=cy+dy;
                if(ry<0||ry>=H_orig){sad=(long)1<<30;break;}
                for(int c=0;c<bs;c++){
                    int cx=bx+c;
                    int rx=cx+dx;  /* position in reference */
                    if(rx<0||rx>=W_orig){sad+=256;continue;}
                    /* compare curr(cx) with ref(cx+dx): MV points to source in ref */
                    sad+=abs(curr[(size_t)ry*W_orig+cx]-hp_ref[(size_t)(ry*2)*W2+(rx*2)]);
                }
            }
            if(sad<best_sad){best_sad=sad;best_dx=dx*4;best_dy=dy*4;}
        }

    /* half-pel refinement around best integer position */
    int base_dx=best_dx/4,base_dy=best_dy/4;
    for(int ddy=-1;ddy<=1;ddy++)
        for(int ddx=-1;ddx<=1;ddx++){
            int qdx=base_dx*4+ddx*2; /* half-pel = 2 quarter-pels */
            int qdy=base_dy*4+ddy*2;
            
            long sad=0;
            for(int r=0;r<bs;r++){
                int cy=by+r;
                for(int c=0;c<bs;c++){
                    int cx=bx+c;
                    uint8_t pred=wubu_sp_get(hp_ref,W2,H2,
                        cx+base_dx,cy+base_dy,(ddx+2)%4,(ddy+2)%4);
                    sad+=abs(curr[cy*(W2/2)+cx]-pred);
                }
            }
            if(sad<best_sad){best_sad=sad;best_dx=qdx;best_dy=qdy;}
        }

    /* quarter-pel refinement */
    for(int ddy=-1;ddy<=1;ddy++)
        for(int ddx=-1;ddx<=1;ddx++){
            int qdx=best_dx+ddx;
            int qdy=best_dy+ddy;
            if(qdx==best_dx&&qdy==best_dy)continue;
            
            long sad=0;
            for(int r=0;r<bs;r++){
                int cy=by+r;
                for(int c=0;c<bs;c++){
                    int cx=bx+c;
                    uint8_t pred=wubu_sp_get(hp_ref,W2,H2,
                        cx+qdx/4,cy+qdy/4,qdx%4,qdy%4);
                    sad+=abs(curr[cy*(W2/2)+cx]-pred);
                }
            }
            if(sad<best_sad){best_sad=sad;best_dx=qdx;best_dy=qdy;}
        }

    *out_dx=best_dx;
    *out_dy=best_dy;
    return best_sad;
}
