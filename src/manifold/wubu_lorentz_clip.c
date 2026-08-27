/*
 * wubu_lorentz_clip.c -- GAP-C042: Lorentz-model CLIP similarity
 * (the C015 queue item — Lorentz variant of manifold CLIP)
 *
 * Research source: MERU/HyCoCLIP lineage (Desai et al.) + "Intriguing
 * Properties of Hyperbolic Embeddings in Vision-Language Models".
 *
 * On the hyperboloid (Lorentz model), similarity between embeddings is
 * the NEGATIVE Lorentzian distance:
 *   sim(a,b) = -d_L(a,b)
 * where d_L uses the Minkowski inner product <x,y>_L = x0*y0 - Σxk*yk:
 *   d_L = arccosh(-<x,y>_L)  for upper-sheet points.
 *
 * vs the Poincaré C002/C013 path: Lorentz gives numerically stabler
 * gradients far from origin (no conformal factor blowup), which matters
 * when CLIP embeddings live near the boundary of the ball.
 */
#include "wubu_lorentz_clip.h"
#include <math.h>
#include <string.h>

float wubu_lc_minkowski_ip(const float* a,const float* b,int D){
    /* <a,b>_L = -a[0]*b[0] + Σ_{k=1..D-1} a[k]*b[k]
     * (mostly-plus convention: L(x,x) = -1/c on the hyperboloid) */
    double ip=-(double)a[0]*b[0];
    for(int d=1;d<D;d++)ip+=(double)a[d]*b[d];
    return (float)ip;
}

float wubu_lc_distance(const float* a,const float* b,int D){
    float ip=wubu_lc_minkowski_ip(a,b,D);
    /* d_L = arccosh(-<a,b>_L): with mostly-plus convention, <a,b> <= -1
     * for timelike-separated upper-sheet points, so -<a,b> >= 1. */
    float nip=-ip;
    if(nip<1.0f)nip=1.0f;
    return acoshf(nip);
}

/* lift an Euclidean vector to the hyperboloid: L(x) with L(x,x)=-1/c,
 * time component set from the spatial norm */
void wubu_lc_lift(const float* v,int Ds,float c,float* out){
    /* out[0] = sqrt(1/c + |v|²); out[1..Ds]=v */
    float n2=0;
    for(int d=0;d<Ds;d++)n2+=v[d]*v[d];
    out[0]=sqrtf(1.0f/c+n2);
    for(int d=0;d<Ds;d++)out[d+1]=v[d];
}

/* project back to hyperboloid: fix the time component so L(x,x)=-1/c */
void wubu_lc_project(float* p,int Ds,float c){
    float n2=0;
    for(int d=1;d<=Ds;d++)n2+=p[d]*p[d];
    p[0]=sqrtf(1.0f/c+n2);
}
