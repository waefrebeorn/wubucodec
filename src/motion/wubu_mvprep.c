/*
 * wubu_mvprep.c -- GROUP 1: Skip mode + Merge mode + MV prediction
 *
 * Skip mode: when the residual is zero after MC, just signal "skip"
 * and copy the MV from a neighbor. Costs 1 bit instead of full MV+residual.
 *
 * Merge mode: reuse the MV of an already-coded neighbor block without
 * transmitting a new MV. The merge candidate list is built from:
 *   A1: left neighbor, B1: above neighbor, B0: above-right,
 *   A0: below-left, T: temporal collocated
 *
 * AMVP (Advanced Motion Vector Prediction): transmit only the DIFFERENCE
 * between the actual MV and the best predictor, not the full MV.
 */
#include "wubu_mvprep.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Skip Mode Detection ===== */

/* Check if a block can be coded as SKIP (zero residual with copied MV) */
int wubu_skip_detect(const uint8_t* curr,const uint8_t* predicted,
                      int W,int H,int bx,int by,int bs,long threshold){
    long sad=0;
    for(int r=0;r<bs;r++){
        int y=by+r;if(y>=H)break;
        for(int c=0;c<bs;c++){
            int x=bx+c;if(x>=W)break;
            sad+=abs(curr[(size_t)y*W+x]-predicted[(size_t)y*W+x]);
            if(sad>threshold)return 0; /* early exit — too much residual */
        }
    }
    return 1; /* SAD below threshold → skip is good enough */
}

/* ===== Merge Candidate List ===== */

int wubu_merge_candidates(const int16_t* mv_field,uint8_t* avail_map,
                           int blocks_per_row,int block_idx,
                           int16_t* out_candidates){
    int count=0;
    
    /* spatial neighbors: left(A1), above(B1), above-right(B0), below-left(A0) */
    int neighbors[4];
    neighbors[0]=block_idx-1;           /* left */
    neighbors[1]=block_idx-blocks_per_row; /* above */
    neighbors[2]=block_idx-blocks_per_row+1; /* above-right */
    neighbors[3]=block_idx+blocks_per_row-1; /* below-left */

    /* availability check */
    uint8_t masks[4]={0x01,0x02,0x04,0x08};
    for(int n=0;n<4;n++){
        if(avail_map[block_idx]&masks[n]){
            /* add this candidate if not duplicate */
            int dup=0;
            for(int i=0;i<count;i++){
                if(out_candidates[i*2]==mv_field[neighbors[n]*2]&&
                   out_candidates[i*2+1]==mv_field[neighbors[n]*2+1]){
                    dup=1;break;
                }
            }
            if(!dup&&count<5){
                out_candidates[count*2]=mv_field[neighbors[n]*2];
                out_candidates[count*2+1]=mv_field[neighbors[n]*2+1];
                count++;
            }
        }
    }

    /* zero MV padding if fewer than needed */
    while(count<5){
        out_candidates[count*2]=0;
        out_candidates[count*2+1]=0;
        count++;
    }
    return count;
}

/* ===== Advanced Motion Vector Prediction ===== */

/* Select best MVP from candidates by minimizing |actual - candidate| */
void wubu_amvp_select(const int16_t* actual_mv,
                       const int16_t* candidates,int n_candidates,
                       int* best_idx,int16_t* mvd_out){
    long best_dist=~(long)0;
    *best_idx=0;
    
    for(int i=0;i<n_candidates;i++){
        long dx=abs(actual_mv[0]-candidates[i*2]);
        long dy=abs(actual_mv[1]-candidates[i*2+1]);
        long dist=dx*dx+dy*dy;  /* squared distance */
        if(dist<best_dist){best_dist=dist;*best_idx=i;}
    }

    /* MVD = actual MV minus the selected predictor */
    mvd_out[0]=actual_mv[0]-candidates[*best_idx*2];
    mvd_out[1]=actual_mv[1]-candidates[*best_idx*2+1];
}

/* Reconstruct the actual MV at decoder side: MV = MVP + MVD */
void wubu_amvp_reconstruct(const int16_t* mvp,const int16_t* mvd,
                            int16_t* mv_out){
    mv_out[0]=mvp[0]+mvd[0];
    mv_out[1]=mvp[1]+mvd[1];
}
