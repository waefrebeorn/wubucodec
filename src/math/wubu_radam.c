/*
 * wubu_radam.c -- GAP-C038: Riemannian Adam on the Poincaré ball
 *
 * Research source: Bécigneul & Ganea arXiv:1810.00760 (Riemannian
 * Adaptive Optimization Methods) — the geoopt RAdam reference.
 *
 * Adam in the tangent space at the current point:
 *   1. rgrad = λ_x²-scaled gradient (Riemannian gradient)
 *   2. m_t = β1·m + (1-β1)·rgrad      (first moment)
 *   3. v_t = β2·v + (1-β2)·rgrad²     (second moment)
 *   4. bias-corrected m̂, v̂
 *   5. update direction u = -m̂/(√v̂+ε)
 *   6. step: x' = exp_x(u) via Möbius add — ON-MANIFOLD by construction
 *
 * vs C023's plain RSGD: adaptive per-coordinate scaling converges much
 * faster when gradient magnitudes vary wildly (which they do near the
 * boundary where λ blows up — v_t absorbs that automatically).
 */
#include "wubu_radam.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

int wubu_radam_init(WubuRADAM* o,int D,float lr,float beta1,float beta2,
                    float eps_c){
    if(D<1)return -1;
    o->D=D;o->t=0;
    o->lr=lr;o->beta1=beta1;o->beta2=beta2;
    o->m=calloc((size_t)D,sizeof(float));
    o->v=calloc((size_t)D,sizeof(float));
    if(!o->m||!o->v)return -2;
    return 0;
}
void wubu_radam_free(WubuRADAM* o){
    free(o->m);free(o->v);
    o->m=NULL;o->v=NULL;
}

/* one Adam step: x updated IN PLACE through exp0(-step).
 * grad is the EUCLIDEAN gradient of the loss wrt x; we convert to the
 * Riemannian gradient by dividing by lambda_x² (the metric pullback). */
void wubu_radam_step(WubuRADAM* o,float* x,const float* egrad,float c){
    int D=o->D;
    /* Riemannian gradient scaling: divide by λ (not λ²) — the practical
     * geoopt form that avoids vanishing steps near center while keeping
     * boundary damping. */
    float n2=0;
    for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float denom=(1-c*n2);
    if(denom<1e-4f)denom=1e-4f;
    float lam=2/denom;

    o->t++;
    /* moments in tangent space */
    float mhat[64],vhat[64],u[64];
    int dd=D<64?D:64;
    for(int d=0;d<dd;d++){
        float rg=egrad[d]/lam;
        o->m[d]=o->beta1*o->m[d]+(1-o->beta1)*rg;
        o->v[d]=o->beta2*o->v[d]+(1-o->beta2)*rg*rg;
        float bc1=1-powf(o->beta1,(float)o->t);
        float bc2=1-powf(o->beta2,(float)o->t);
        mhat[d]=o->m[d]/bc1;
        vhat[d]=o->v[d]/bc2;
        u[d]=-o->lr*mhat[d]/(sqrtf(vhat[d])+1e-8f);
    }
    if(dd<D)memset(u+dd,0,sizeof(float)*(size_t)(D-dd));

    /* exponential map step: x' = exp0(x_shift) where shift moves along -u.
     * For small steps, exp0(x - u) ≈ x ⊕ (-u). We use mobius_add directly:
     * x' = x ⊕ (-u_scaled). */
    float uu=0,vv=0,uv=0;
    float neg[64];
    /* u already points DOWNHILL (it's -lr*m̂); use it directly */
    for(int d=0;d<dd;d++){neg[d]=u[d];uu+=x[d]*x[d];vv+=neg[d]*neg[d];uv+=x[d]*neg[d];}
    float num1=1+2*c*uv+c*vv,num2=1-c*uu;
    float mden=1+2*c*uv+c*c*uu*vv;
    if(mden<1e-10f)mden=1e-10f;
    for(int d=0;d<dd;d++)x[d]=(num1*x[d]+num2*neg[d])/mden;
    for(int d=dd;d<D;d++)x[d]=0;

    /* fp32 boundary cap */
    float n2c=0;for(int d=0;d<D;d++)n2c+=x[d]*x[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)x[d]*=s;}
}

/* batch convenience */
void wubu_radam_batch(WubuRADAM* o,float* xs,int N,const float* grads,
                      int D,float c){
    for(int i=0;i<N;i++)
        wubu_radam_step(o,xs+(size_t)i*D,grads+(size_t)i*D,c);
}
