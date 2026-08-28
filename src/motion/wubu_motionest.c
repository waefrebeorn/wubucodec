/*
 /* wubu_motionest.c -- GAP-C075: Block matching motion estimation
  * with hexagon integer search and half-then-quarter-pel refinement.
  *
  * Pipeline:
  *   1. Integer-pel hexagon search (6 directions, better than diamond)
  *   2. Half-pel refinement around best integer MV (4 positions)
  *   3. Quarter-pel refinement around best half-pel (4 positions)
  *   4. Bilinear interpolation for sub-pel positions
  *
  * Research: x264's uneven multi-hexagon search achieves within 0.5% of
  * full exhaustive search. Half-then-quarter refinement is ~4x faster
  * than full quarter-pel search with negligible quality loss.
  */
#include "wubu_motionest.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

/* SAD between a block in curr at (bx,by) and ref at (bx+dx,by+dy) */
/* Uses SIMD for in-bounds case, scalar for boundary check */
long wubu_me_sad(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,int dx,int dy){
    /* Fast path: fully in-bounds */
    int rx = bx + dx;
    int ry = by + dy;
    if(rx >= 0 && ry >= 0 && rx + bs <= W && ry + bs <= H){
        long sad = 0;
        for(int r = 0; r < bs; r++){
            const uint8_t* cp = curr + (by + r) * W + bx;
            const uint8_t* rp = ref + (ry + r) * W + rx;
            int c = 0;
#if defined(__AVX2__)
            /* Process 8 rows, 8 cols each = 64 bytes per row */
            for(; c + 8 <= bs; c += 8){
                __m128i vc = _mm_loadl_epi64((const __m128i*)(cp + c));
                __m128i vr = _mm_loadl_epi64((const __m128i*)(rp + c));
                __m128i sad8 = _mm_sad_epu8(vc, vr);
                sad += _mm_extract_epi64(sad8, 0);
            }
#endif
            for(; c < bs; c++){
                sad += abs((int)cp[c] - (int)rp[c]);
            }
        }
        return sad;
    }
    
    /* Slow path: boundary check per pixel */
    long sad = 0;
    for(int r = 0; r < bs; r++){
        int cy = by + r;
        int ry2 = cy + dy;
        if(ry2 < 0 || ry2 >= H) return 1LL << 30;
        for(int c = 0; c < bs; c++){
            int cx = bx + c;
            int rx2 = cx + dx;
            if(rx2 < 0 || rx2 >= W){sad += (1 << 20); continue;}
            sad += abs(curr[cy * W + cx] - ref[ry2 * W + rx2]);
        }
    }
    return sad;
}

/* Bilinearly interpolated SAD for quarter-pel positions.
 * dx,dy are in quarter-pel units (e.g., dx=1 means 0.25 pel, dx=2 means 0.5 pel).
 * NOTE: Single-plane (Y-only) — no *3 multiplier on index */
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
            /* Bilinear interpolation (single-plane, no *3) */
            if(ry0<0||ry0>=H||rx0<0||rx0>=W){sad+=(1<<20);continue;}
            int v00 = ref[ry0*W+rx0];
            int v10 = (rx0+1<W) ? ref[ry0*W+rx0+1] : v00;
            int v01 = (ry0+1<H) ? ref[(ry0+1)*W+rx0] : v00;
            int v11 = (rx0+1<W&&ry0+1<H) ? ref[(ry0+1)*W+rx0+1] : v00;
            float v = (float)v00*(1.0f-fx)*(1.0f-fy) +
                      (float)v10*fx*(1.0f-fy) +
                      (float)v01*(1.0f-fx)*fy +
                      (float)v11*fx*fy;
            int v_int = (int)(v+0.5f);
            if(v_int<0)v_int=0; if(v_int>255)v_int=255;
            sad+=abs(curr[cy*W+cx]-v_int);
        }
    }
    return sad;
}

/* find best motion vector for one block using hexagon + half-then-quarter-pel refinement */
long wubu_me_block(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,
                    int search_range,int* out_dx,int* out_dy){
    /* Phase 1: Integer-pel hexagon search (6 directions) */
    int best_dx=0,best_dy=0;
    long best_sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,0,0);

    /* Hexagon pattern: 6 directions (4 cardinal + 2 diagonal) */
    static const int hex_dx[]={0,-1,1,-1,1,0};
    static const int hex_dy[]={-1,0,0,1,1,-1};

    int improved=1;
    while(improved){
        improved=0;
        for(int d=0;d<6;d++){
            int ndx=best_dx+hex_dx[d];
            int ndy=best_dy+hex_dy[d];
            if(abs(ndx)>search_range||abs(ndy)>search_range)continue;
            long sad=wubu_me_sad(curr,ref,W,H,bx,by,bs,ndx,ndy);
            if(sad<best_sad){
                best_sad=sad;best_dx=ndx;best_dy=ndy;
                improved=1;
            }
        }
    }

    /* Phase 2: Half-pel refinement (4 positions around best integer) */
    /* Half-pel positions: (0.5,0), (0,0.5), (0.5,0.5), (-0.5,0.5) */
    int best_hx=0, best_hy=0;
    long best_sad_half=best_sad;
    static const int half_dx[]={2,0,2,-2};
    static const int half_dy[]={0,2,2,2};
    for(int d=0;d<4;d++){
        int dx_qpel = best_dx*4 + half_dx[d];
        int dy_qpel = best_dy*4 + half_dy[d];
        long sad=wubu_me_sad_qpel(curr,ref,W,H,bx,by,bs,dx_qpel,dy_qpel);
        if(sad<best_sad_half){
            best_sad_half=sad;
            best_hx=half_dx[d];
            best_hy=half_dy[d];
        }
    }

    /* Phase 3: Quarter-pel refinement around best half-pel (4 positions) */
    int best_qx=best_hx, best_qy=best_hy;
    long best_sad_qpel=best_sad_half;
    /* Quarter-pel offsets around half-pel: ±1 in each direction */
    static const int qpel_dx[]={-1,1,0,0};
    static const int qpel_dy[]={0,0,-1,1};
    for(int d=0;d<4;d++){
        int dx_qpel = best_dx*4 + best_hx + qpel_dx[d];
        int dy_qpel = best_dy*4 + best_hy + qpel_dy[d];
        long sad=wubu_me_sad_qpel(curr,ref,W,H,bx,by,bs,dx_qpel,dy_qpel);
        if(sad<best_sad_qpel){
            best_sad_qpel=sad;
            best_qx=best_hx+qpel_dx[d];
            best_qy=best_hy+qpel_dy[d];
        }
    }

    *out_dx = best_dx*4 + best_qx;
    *out_dy = best_dy*4 + best_qy;
    return best_sad_qpel;
}

/* estimate motion vectors for all blocks in a frame */
int wubu_me_frame(const uint8_t* curr,const uint8_t* ref,
                   int W,int H,int bs,int search_range,
                   int* out_mvs /* [n_blocks_x*n_blocks_y][2], in quarter-pel units */){
    int nbx=W/bs,nby=H/bs;
    int count=0;
    /* Use OpenMP for large frames only (overhead dominates for small frames) */
    #pragma omp parallel for schedule(dynamic, 4) if(nbx*nby > 64)
    for(int by=0;by<nby;by++)
        for(int bx=0;bx<nbx;bx++){
            int dx,dy;
            wubu_me_block(curr,ref,W,H,bx*bs,by*bs,bs,search_range,&dx,&dy);
            int idx = by*nbx+bx;
            out_mvs[idx*2]=dx;
            out_mvs[idx*2+1]=dy;
            count++;
        }
    return count;
}

/* apply motion compensation: build predicted frame from reference + MVs.
 * MVs are in quarter-pel units (divide by 4.0 to get actual offset).
 * NOTE: Single-plane (Y-only) — no *3 multiplier on index */
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
                        predicted[sy*W+sx]=0;
                        continue;
                    }
                    int v00 = ref[ry0*W+rx0];
                    int v10 = (rx0+1<W) ? ref[ry0*W+rx0+1] : v00;
                    int v01 = (ry0+1<H) ? ref[(ry0+1)*W+rx0] : v00;
                    int v11 = (rx0+1<W&&ry0+1<H) ? ref[(ry0+1)*W+rx0+1] : v00;
                    float v = (float)v00*(1.0f-fx)*(1.0f-fy) +
                              (float)v10*fx*(1.0f-fy) +
                              (float)v01*(1.0f-fx)*fy +
                              (float)v11*fx*fy;
                    int v_int = (int)(v+0.5f);
                    if(v_int<0)v_int=0; if(v_int>255)v_int=255;
                    predicted[sy*W+sx]=(uint8_t)v_int;
                }
            }
        }
}
