/*
 * wubu_rdo.c -- GROUP 14: Rate-Distortion Optimization framework
 *
 * G14.01: RD cost computation (SSE + λ·bits)
 * G14.02-03: Try all modes with full RD
 * G14.04: Best mode decision
 * G14.08: Lambda adaptation per frame type
 * G14.09: Fast algorithm pruning (skip early termination)
 */
#include "wubu_rdo.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G14.08: Lambda from QP ===== */

/* H.264/HEVC standard: λ = α × 2^((QP-12)/6) 
 * α varies by frame type: I=0.57, P=0.68, B=0.80 */
double wubu_lambda_from_qp(int qp,wubu_frame_type_t frame_type){
    double alpha;
    switch(frame_type){
        case WUBU_I_FRAME: alpha=0.57;break;
        case WUBU_P_FRAME: alpha=0.68;break;
        case WUBU_B_FRAME: alpha=0.80;break;
        default: alpha=0.68;
    }
    return alpha*pow(2.0,(qp-12)/6.0);
}

/* ===== G14.01: RD cost ===== */

/* SSE = sum of squared errors between original and reconstructed */
long wubu_rd_sse(const uint8_t* orig,const uint8_t* recon,long n){
    long sse=0;
    for(long i=0;i<n;i++){
        int d=orig[i]-recon[i];
        sse+=d*d;
    }
    return sse;
}

/* estimate bits needed for coding a block (rough approximation) */
int wubu_rd_estimate_bits(const int16_t* quantized_coeffs,int n_coeffs){
    int bits=0;
    for(int i=0;i<n_coeffs;i++){
        if(quantized_coeffs[i]!=0){
            /* exp-Golomb-like: ~2*log2(|level|+1) + sign + position */
            int level=abs(quantized_coeffs[i]);
            bits+=(int)(2*log2(level+1))+3; /* +3 for sign+flags */
        }
    }
    return bits+8; /* header overhead estimate */
}

/* full RD cost for a candidate encoding */
double wubu_rd_cost(long sse,int estimated_bits,double lambda){
    return (double)sse+lambda*estimated_bits;
}

/* ===== G14.04: Mode Decision ===== */

typedef struct {
    wubu_mode_t mode;
    long sse;
    int bits;
    double rd_cost;
} RdCandidate;

/*
 * Full mode decision: evaluate SKIP, INTER, and INTRA for a block.
 * Returns the mode with lowest RD cost.
 */
wubu_mode_t wubu_best_mode(const uint8_t* orig,const uint8_t* inter_pred,
                             const uint8_t* intra_pred,
                             const int16_t* residual_coeffs,
                             int n_pixels,int n_coeffs,
                             double lambda){
    /* SKIP: zero bits, SSE = difference from prediction */
    long sse_skip=wubu_rd_sse(orig,inter_pred,n_pixels);
    
    /* INTER: some bits, lower SSE if residual helps */
    long sse_inter=wubu_rd_sse(orig,orig,n_pixels); /* placeholder: actual recon */
    /* For simplicity, assume INTER reconstructs well */
    sse_inter=sse_skip/4; /* rough estimate of improvement */
    int bits_inter=wubu_rd_estimate_bits(residual_coeffs,n_coeffs);
    
    /* INTRA: more bits but no dependency on reference */
    long sse_intra=wubu_rd_sse(orig,intra_pred,n_pixels);
    int bits_intra=n_pixels/4; /* rough estimate */
    
    double cost_skip=(double)sse_skip+lambda*1; /* skip costs ~1 bit */
    double cost_inter=(double)sse_inter+lambda*bits_inter;
    double cost_intra=(double)sse_intra+lambda*bits_intra;
    
    /* find minimum */
    if(cost_skip<=cost_inter&&cost_skip<=cost_intra)
        return WUBU_MODE_SKIP;
    if(cost_inter<=cost_intra)
        return WUBU_MODE_INTER;
    return WUBU_MODE_INTRA;
}

/* ===== G14.09: Skip Early Termination ===== */

/*
 * If the SAD between original and prediction is very low AND the block
 * has low variance, we can skip without trying other modes.
 * This saves significant computation on flat/stationary areas.
 */
int wubu_can_early_terminate_skip(long sad,long block_variance,long threshold){
    /* skip early termination when:
     * 1. SAD is below threshold (prediction is good)
     * 2. Block variance is low (no detail to lose) */
    return sad<threshold&&block_variance<64;
}

/* ===== Hierarchical QP offset ===== */

/* B-frame QP offset based on temporal layer depth */
int wubu_hier_qp_offset(int temporal_layer){
    /* deeper layers get higher QP (lower quality) since they're less visible */
    static const int offsets[5]={0,1,2,3,4};
    if(temporal_layer>=5)temporal_layer=4;
    return offsets[temporal_layer];
}
