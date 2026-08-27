/*
 * wubu_rdomode.c -- GAP-C029: Rate-distortion mode decision (HEVC-style)
 *
 * The inter-prediction mode chooser from HEVC/VVC: for each candidate
 * mode (SKIP / MERGE / AMVP), compute J = D + λ·R and pick the minimum.
 *
 * For the WuBu codec: modes are
 *   SKIP      — zero residual, copy prediction (rate = flag only)
 *   RESIDUAL  — quantized residual in IR band (rate = bits from B010)
 *   INPAINT   — flow-only prediction, no residual, no MV (rate = flag)
 * Distortion = geodesic distance between reconstructed and original latents.
 */
#include "wubu_rdomode.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

float wubu_rd_geodesic(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];
        ab2+=df*df;
        a2+=a[d]*a[d];
        b2+=b[d]*b[d];
    }
    float den=(1.0f-c*a2)*(1.0f-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1.0f+2.0f*c*ab2/den;
    return acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
}

/* lambda from qp (quantization parameter): λ = 0.85 * 2^((qp-12)/6) — HEVC */
float wubu_rd_lambda(int qp){
    return 0.85f*powf(2.0f,(float)(qp-12)/6.0f);
}

WubuRDMode wubu_rd_decide(const float* orig,const float* pred,
                           const float* recon_residual,
                           int D,float c,float lambda,
                           float rate_skip,float rate_residual){
    /* Mode SKIP: reconstruction == prediction */
    float d_skip=wubu_rd_geodesic(orig,pred,D,c);
    float j_skip=d_skip*d_skip+lambda*rate_skip;

    /* Mode RESIDUAL: reconstruction = pred + residual (already computed) */
    float d_res=wubu_rd_geodesic(orig,recon_residual,D,c);
    float j_res=d_res*d_res+lambda*rate_residual;

    WubuRDMode m;
    if(j_skip<=j_res){
        m.mode=WUBU_RD_SKIP;
        m.cost=j_skip;
        m.distortion=d_skip;
        m.bits=(int)rate_skip;
    }else{
        m.mode=WUBU_RD_RESIDUAL;
        m.cost=j_res;
        m.distortion=d_res;
        m.bits=(int)rate_residual;
    }
    return m;
}
