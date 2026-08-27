/*
 * wubu_fastmode.c -- Fast mode decision & CU depth pruning
 *
 * Speed optimizations that skip unnecessary computation without
 * significantly hurting quality. These are what make x264 "veryfast"
 * vs "placebo" — same output quality at 10-100x speed.
 *
 * 1. Early SKIP: if SAD is tiny, skip all other modes
 * 2. Depth decision: if parent CU RD cost is already low, don't split
 * 3. Mode pruning: if INTRA cost >> INTER, skip intra search
 * 4. AQ: adaptive quantization — lower QP for detail areas
 */
#define M_PI 3.14159265358979f
#include "wubu_fastmode.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Early SKIP decision ===== */

int wubu_fm_early_skip(long sad,long variance,long sad_threshold){
    /* two conditions must both hold:
     * 1. prediction error below threshold
     * 2. block has low detail (variance) */
    return sad<sad_threshold&&variance<256;
}

/* ===== CU Depth Decision ===== */

/*
 * Decide whether to split a block into 4 sub-blocks.
 * Split only if the parent's RD cost is much worse than the sum
 * of children's estimated costs.
 *
 * Returns 1 = split (search sub-blocks), 0 = don't split.
 */
int wubu_fm_should_split(double rd_parent,double rd_children_sum,
                           double lambda,int n_children){
    /* splitting costs extra bits for partition info + child MVs */
    double split_overhead=lambda*(4+n_children); /* ~4 bits header + MVs */
    
    /* split if children are sufficiently better */
    double improvement=rd_parent-(rd_children_sum+split_overhead);
    return improvement>lambda*8; /* require meaningful gain */
}

/* estimate children's combined cost without actually encoding them */
double wubu_fm_estimate_split_cost(const uint8_t* orig,const uint8_t* pred,
                                     int W,int H,int bx,int by,int bs,
                                     double lambda){
    /* compute per-quadrant SAD and sum */
    long total_sad=0;
    int half=bs/2;
    for(int q=0;q<4;q++){
        int qx=bx+(q%2)*half;
        int qy=by+(q/2)*half;
        long quad_sad=0;
        for(int r=0;r<half;r++)
            for(int c=0;c<half;c++){
                int py=qy+r,px=qx+c;
                if(py<H&&px<W)
                    quad_sad+=abs(orig[(size_t)py*W+px]-pred[(size_t)py*W+px]);
            }
        /* each quadrant needs its own MV (~4 bits) */
        total_sad+=quad_sad;
    }
    /* estimated cost = SSE + λ·(4 MVs × 4 bits + partition overhead) */
    return (double)total_sad*total_sad/(bs*bs)+lambda*(16+4);
}

/* ===== Intra mode pruning ===== */

/* Skip intra search if inter prediction is already very good */
int wubu_fm_skip_intra(long sad_inter,long sad_intra_threshold){
    return sad_inter<sad_intra_threshold;
}

/* ===== Adaptive Quantization (AQ) ===== */

/*
 * Measure local activity and adjust QP accordingly:
 * - High detail → lower QP (more bits to preserve)
 * - Low detail → higher QP (bits won't be missed)
 */
int wubu_aq_offset(const uint8_t* img,int W,int H,
                     int bx,int by,int bs,int max_offset){
    /* compute local variance */
    long sum=0,sum_sq=0,n=0;
    for(int r=0;r<bs;r++)
        for(int c=0;c<bs;c++){
            int y=by+r,x=bx+c;
            if(y>=H||x>=W)continue;
            long v=img[(size_t)y*W+x];
            sum+=v;sum_sq+=v*v;n++;
        }
    if(n==0)return 0;
    
    long mean=sum/n;
    long variance=sum_sq/n-mean*mean;
    
    /* map variance to QP offset */
    if(variance>2000)return -max_offset;   /* very detailed */
    if(variance>500)return -max_offset/2;  /* moderately detailed */
    if(variance>100)return 0;               /* normal */
    if(variance>20)return max_offset/2;     /* smooth */
    return max_offset;                       /* very flat */
}

/* ===== Combined fast-mode pipeline ===== */

FmDecision wubu_fm_decide(long sad,long variance,long skip_threshold,
                            long intra_threshold){
    /* Stage 1: can we skip everything? */
    if(wubu_fm_early_skip(sad,variance,skip_threshold))
        return FM_DECISION_SKIP;
    
    /* Stage 2: is inter good enough to skip intra? */
    if(sad<intra_threshold)
        return FM_DECISION_INTER_ONLY;
    
    /* Stage 3: need full search */
    return FM_DECISION_FULL_SEARCH;
}
