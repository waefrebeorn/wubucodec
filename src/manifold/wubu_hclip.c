/*
 * wubu_hclip.c -- GAP-C046: Hyperbolic gradient clipping
 * (conformal-aware tangent-norm clipping)
 *
 * Gradient clipping (Pascanu et al. 2013) adapted to the ball: the
 * gradient of a loss wrt an on-ball point blows up near the boundary
 * because the conformal factor lambda = 2/(1-c|x|^2) -> inf. Naive
 * Euclidean clipping under-clips boundary gradients and over-clips
 * center gradients.
 *
 * Correct form: convert to the RIEMANNIAN gradient g/lambda^2, clip THAT
 * to a max tangent norm, then rescale back. This equalizes effective step
 * size across the whole manifold — the same property C023's RSGD guard
 * provides per-step, but as a reusable pre-optimizer pass.
 */
#include "wubu_hclip.h"
#include <math.h>

float wubu_hc_riemannian_norm(const float* egrad,const float* x,
                               int D,float c){
    /* |g|_R = lambda * |g|_E with lambda=2/(1-c|x|^2) */
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float denom=1-c*n2;
    if(denom<1e-4f)denom=1e-4f;
    float lam=2.0f/denom;
    float en2=0;
    for(int d=0;d<D;d++)en2+=egrad[d]*egrad[d];
    return lam*sqrtf(en2);
}

int wubu_hc_clip(float* grad,const float* x,int D,float c,
                  float max_rnorm){
    if(max_rnorm<=0)return -1;
    float rn=wubu_hc_riemannian_norm(grad,x,D,c);
    if(rn<=max_rnorm)return 0;   /* no clipping needed */
    /* scale euclidean grad so riemannian norm == max_rnorm */
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float denom=1-c*n2;
    if(denom<1e-4f)denom=1e-4f;
    float lam=2.0f/denom;
    float target_en=max_rnorm/lam;
    float en=rn/lam;          /* euclidean norm: rn = lam*en */
    if(en<1e-12f)return 0;
    float f=target_en/en;
    for(int d=0;d<D;d++)grad[d]*=f;
    return 1;   /* clipped */
}
