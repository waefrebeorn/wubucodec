/*
 * wubu_lrsched.c -- GAP-C049: Learning-rate schedules with conformal
 * damping (warmup+cosine, hyperbolically damped)
 *
 * The standard warmup-linear-cosine schedule (LLM training standard),
 * plus a HYPERBOLIC DAMPING factor: the scheduled LR is divided by the
 * average conformal factor lambda(x) of the parameters being optimized.
 * Points near the boundary automatically get smaller steps — schedule
 * and geometry working together instead of fighting.
 */
#include "wubu_lrsched.h"
#include <math.h>

float wubu_lr_warmup_cosine(int t,int T_warmup,int T_total,
                             float eta_max,float eta_min){
    if(T_total<=0)return eta_min;
    if(t<0)t=0;
    if(t>=T_total)return eta_min;
    if(t<T_warmup){
        int tw=T_warmup>0?T_warmup:1;
        return eta_max*(float)t/(float)tw;
    }
    float prog=(float)(t-T_warmup)/(float)(T_total-T_warmup);
    if(prog<0)prog=0;
    if(prog>1)prog=1;
    return eta_min+0.5f*(eta_max-eta_min)*(1.0f+cosf(3.14159265f*prog));
}

float wubu_lr_damped(float base_lr,const float* x,int D,float c,
                      float max_damp){
    /* damp = clamp(lambda_avg, 1, max_damp); lr_eff = base/damp */
    if(D<=0)return base_lr;
    double lam_sum=0;
    int dd=D<64?D:64;
    for(int d=0;d<dd&&d<64;d++){
        float n2=x[d]*x[d];
        float denom=1-c*n2;
        if(denom<1e-4f)denom=1e-4f;
        lam_sum+=2.0f/denom;
    }
    float lam=(float)(lam_sum/dd);
    if(lam<1.0f)lam=1.0f;
    if(lam>max_damp)lam=max_damp;
    return base_lr/lam;
}

/* convenience: full schedule + damping in one call */
float wubu_lr_schedule_step(int t,int T_warmup,int T_total,
                             float eta_max,float eta_min,
                             const float* x,int D,float c,
                             float max_damp){
    float lr=wubu_lr_warmup_cosine(t,T_warmup,T_total,eta_max,eta_min);
    return wubu_lr_damped(lr,x,D,c,max_damp);
}
