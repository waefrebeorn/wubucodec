/*
 * wubu_hanalogy.c -- GAP-D035: Hyperbolic analogy completion
 * (geodesic offset arithmetic: a - b + c on the ball)
 *
 * The word2vec king−man+woman=queen trick, hyperbolized. On the ball
 * the "difference" between a and b is the TANGENT vector at origin:
 *   diff(a,b) = log0(a) − log0(b)
 * and analogy prediction applies that offset to c:
 *   answer = exp0( log0(c) + diff(a,b) )
 * then nearest-neighbor lookup over the vocabulary by geodesic distance.
 *
 * Tangent-space offsets are the correct generalization: they're
 * translation-invariant (the same relation applies anywhere on the
 * ball), which raw coordinate subtraction is not.
 */
#include "wubu_hanalogy.h"
#include <math.h>
#include <string.h>

static float an_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

/* log_0 into out */
void wubu_ha_log0(const float* x,int D,float c,float* out){
    float n2=0;
    for(int d=0;d<D&&d<512;d++)n2+=x[d]*x[d];
    float nx=sqrtf(n2);
    if(nx>1e-10f){
        float arg=sqrtf(c)*nx;
        if(arg>0.999999f)arg=0.999999f;
        float zn=atanhf(arg)/(sqrtf(c)*nx);
        for(int d=0;d<D&&d<512;d++)out[d]=zn*x[d];
    }else{
        memset(out,0,sizeof(float)*(size_t)D);
    }
}

int wubu_ha_analogy(const float* emb,int vocab,int D,float c,
                     int a,int b,int cc,unsigned* seed){
    if(a<0||b<0||cc<0||a>=vocab||b>=vocab||cc>=vocab)return -1;

    float la[512],lb[512],lc[512],target[512];
    wubu_ha_log0(emb+(size_t)a*D,D,c,la);
    wubu_ha_log0(emb+(size_t)b*D,D,c,lb);
    wubu_ha_log0(emb+(size_t)cc*D,D,c,lc);
    int dd=D<512?D:512;
    for(int d=0;d<dd&&d<512;d++)
        target[d]=lc[d]+la[d]-lb[d];

    /* exp_0(target) */
    float vn2=0;
    for(int d=0;d<dd&&d<512;d++)vn2+=target[d]*target[d];
    float nv=sqrtf(vn2);
    float pred[64];
    if(nv>1e-10f){
        float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
        for(int d=0;d<dd&&d<64;d++)pred[d]=coeff*target[d];
    }else{
        memset(pred,0,sizeof(pred));
    }
    /* nearest vocab item excluding inputs */
    int best=-1;float bd=1e30f;
    for(int v=0;v<vocab;v++){
        if(v==a||v==b||v==cc)continue;
        float d=an_dist(pred,emb+(size_t)v*D,D,c);
        if(d<bd){bd=d;best=v;}
    }
    (void)seed;
    return best;
}
