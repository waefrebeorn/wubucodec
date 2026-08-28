/*
 /* wubu_motionest.c -- GAP-C075: Block matching motion estimation
  * with hierarchical coarse-to-fine search and variable block sizes.
  *
  * Pipeline:
  *   1. Hierarchical: search on downsampled frame, refine at full res
  *   2. Integer-pel hexagon search (6 directions)
  *   3. Half-pel refinement (4 positions)
  *   4. Quarter-pel refinement (4 positions)
  *   5. Bilinear interpolation for sub-pel positions
  *
  * Research: Hierarchical ME reduces cost ~4x for large frames (8K).
  * Variable block sizes (16x16, 32x32) crucial for 8K flat regions.
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

/* Downsample frame by 2x (box filter) for hierarchical ME */
static uint8_t* downsample_2x(const uint8_t* src, int W, int H){
    int W2 = W / 2, H2 = H / 2;
    uint8_t* dst = malloc((size_t)W2 * H2);
    for(int y = 0; y < H2; y++){
        for(int x = 0; x < W2; x++){
            int sum = 0;
            for(int dy = 0; dy < 2; dy++)
                for(int dx = 0; dx < 2; dx++)
                    sum += src[(y*2+dy)*W + (x*2+dx)];
            dst[y*W2 + x] = (uint8_t)(sum / 4);
        }
    }
    return dst;
}

/* Hierarchical motion estimation: coarse-to-fine */
long wubu_me_block_hierarchical(const uint8_t* curr, const uint8_t* ref,
                                  int W, int H, int bx, int by, int bs,
                                  int search_range, int* out_dx, int* out_dy){
    /* For small frames or small blocks, use direct search */
    if(W * H < 100000 || bs <= 8){
        return wubu_me_block(curr, ref, W, H, bx, by, bs, search_range, out_dx, out_dy);
    }
    
    /* Downsample both frames by 2x */
    uint8_t* curr_ds = downsample_2x(curr, W, H);
    uint8_t* ref_ds = downsample_2x(ref, W, H);
    int W2 = W / 2, H2 = H / 2;
    
    /* Search on downsampled frame with half the search range */
    int bx_ds = bx / 2, by_ds = by / 2;
    int bs_ds = bs / 2;
    if(bs_ds < 4) bs_ds = 4;
    int dx_coarse, dy_coarse;
    int sr_ds = search_range / 2;
    if(sr_ds < 2) sr_ds = 2;
    
    long coarse_sad = wubu_me_block(curr_ds, ref_ds, W2, H2, bx_ds, by_ds, bs_ds, sr_ds, &dx_coarse, &dy_coarse);
    
    free(curr_ds);
    free(ref_ds);
    
    /* Refine at full resolution around the coarse MV */
    /* Coarse MV is in quarter-pel units of downsampled frame = half-pel of full frame */
    int refine_dx = dx_coarse * 2;  /* Scale to full-res quarter-pel units */
    int refine_dy = dy_coarse * 2;
    
    /* Skip refinement for very flat blocks (coarse SAD is very low) */
    if(coarse_sad < bs_ds * bs_ds * 4){
        *out_dx = refine_dx;
        *out_dy = refine_dy;
        return coarse_sad;
    }
    
    /* Small refinement search (±1 pixel around coarse) for speed */
    int best_dx = refine_dx, best_dy = refine_dy;
    long best_sad = wubu_me_sad(curr, ref, W, H, bx, by, bs, refine_dx, refine_dy);
    
    for(int dy = -1; dy <= 1; dy++){
        for(int dx = -1; dx <= 1; dx++){
            if(dx == 0 && dy == 0) continue;
            int ndx = refine_dx + dx;
            int ndy = refine_dy + dy;
            if(abs(ndx) > search_range || abs(ndy) > search_range) continue;
            long sad = wubu_me_sad(curr, ref, W, H, bx, by, bs, ndx, ndy);
            if(sad < best_sad){
                best_sad = sad;
                best_dx = ndx;
                best_dy = ndy;
            }
        }
    }
    
    /* Quarter-pel refinement */
    int best_hx = 0, best_hy = 0;
    long best_sad_qpel = best_sad;
    static const int half_dx[] = {2, 0, 2, -2};
    static const int half_dy[] = {0, 2, 2, 2};
    for(int d = 0; d < 4; d++){
        int dx_qpel = best_dx * 4 + half_dx[d];
        int dy_qpel = best_dy * 4 + half_dy[d];
        long sad = wubu_me_sad_qpel(curr, ref, W, H, bx, by, bs, dx_qpel, dy_qpel);
        if(sad < best_sad_qpel){
            best_sad_qpel = sad;
            best_hx = half_dx[d];
            best_hy = half_dy[d];
        }
    }
    
    *out_dx = best_dx * 4 + best_hx;
    *out_dy = best_dy * 4 + best_hy;
    return best_sad_qpel;
}

/* find best motion vector for one block using hexagon + half-then-quarter-pel refinement */
long wubu_me_block(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,
                    int search_range,int* out_dx,int* out_dy){
    return wubu_me_block_fast(curr, ref, W, H, bx, by, bs, search_range, out_dx, out_dy, NULL, W/bs);
}

/* Fast ME with diamond search and early termination */
long wubu_me_block_fast(const uint8_t* curr,const uint8_t* ref,
                    int W,int H,int bx,int by,int bs,
                    int search_range,int* out_dx,int* out_dy,
                    const int16_t* mvs,int nbx){
    /* Get MVP from neighbors */
    int mvp_x = 0, mvp_y = 0;
    if(mvs){
        int idx = by/bs * nbx + bx/bs;
        if(idx >= 0 && idx < nbx * (H/bs)){
            mvp_x = mvs[idx*2];
            mvp_y = mvs[idx*2+1];
        }
    }
    
    /* Start from MVP */
    int best_dx = mvp_x;
    int best_dy = mvp_y;
    long best_sad = wubu_me_sad(curr, ref, W, H, bx, by, bs, mvp_x, mvp_y);
    
    /* Early termination: if MVP SAD is very low, skip search */
    if(best_sad < bs * bs * 4){
        *out_dx = best_dx;
        *out_dy = best_dy;
        return best_sad;
    }
    
    /* Also check (0,0) for static regions */
    long sad_zero = wubu_me_sad(curr, ref, W, H, bx, by, bs, 0, 0);
    if(sad_zero < best_sad){
        best_sad = sad_zero;
        best_dx = 0;
        best_dy = 0;
    }
    
    /* Diamond search around best */
    static const int dia_dx[] = {0, -1, 0, 1, 0};
    static const int dia_dy[] = {-1, 0, 1, 0, -1};
    
    int improved = 1;
    while(improved){
        improved = 0;
        for(int d = 0; d < 4; d++){
            int ndx = best_dx + dia_dx[d];
            int ndy = best_dy + dia_dy[d];
            if(abs(ndx) > search_range || abs(ndy) > search_range) continue;
            long sad = wubu_me_sad(curr, ref, W, H, bx, by, bs, ndx, ndy);
            if(sad < best_sad){
                best_sad = sad;
                best_dx = ndx;
                best_dy = ndy;
                improved = 1;
            }
        }
    }
    
    *out_dx = best_dx;
    *out_dy = best_dy;
    return best_sad;
}

/* Full ME with quarter-pel refinement for quality mode */
long wubu_me_block_full(const uint8_t* curr,const uint8_t* ref,
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
                   int* out_mvs){
    int nbx=W/bs,nby=H/bs;
    int count=0;
    /* Use hierarchical ME for large frames, fast ME for small frames */
    if(W * H > 100000){
        /* Hierarchical: coarse-to-fine */
        for(int by=0;by<nby;by++)
            for(int bx=0;bx<nbx;bx++){
                int dx,dy;
                wubu_me_block_hierarchical(curr,ref,W,H,bx*bs,by*bs,bs,search_range,&dx,&dy);
                int idx = by*nbx+bx;
                out_mvs[idx*2]=dx;
                out_mvs[idx*2+1]=dy;
            }
    }else{
        /* Fast ME for small frames */
        for(int by=0;by<nby;by++)
            for(int bx=0;bx<nbx;bx++){
                int dx,dy;
                wubu_me_block(curr,ref,W,H,bx*bs,by*bs,bs,search_range,&dx,&dy);
                int idx = by*nbx+bx;
                out_mvs[idx*2]=dx;
                out_mvs[idx*2+1]=dy;
            }
    }
    return count;
}

/* Fast ME frame - diamond search only, no quarter-pel */
int wubu_me_frame_fast(const uint8_t* curr,const uint8_t* ref,
                        int W,int H,int bs,int search_range,
                        int* out_mvs){
    int nbx=W/bs,nby=H/bs;
    for(int by=0;by<nby;by++)
        for(int bx=0;bx<nbx;bx++){
            int dx,dy;
            wubu_me_block_fast(curr,ref,W,H,bx*bs,by*bs,bs,search_range,&dx,&dy,NULL,nbx);
            int idx = by*nbx+bx;
            out_mvs[idx*2]=dx;
            out_mvs[idx*2+1]=dy;
        }
    return nbx*nby;
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
