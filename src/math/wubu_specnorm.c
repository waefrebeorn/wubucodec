/*
 * wubu_specnorm.c -- GAP-C043: Spectral normalization via power iteration
 * (Lipschitz control for hyperbolic network layers)
 *
 * Research source: Miyato et al. 2018 (spectral normalization) + Zhai
 * ICML 2023 (sigmaReparam — ties into B017's entropy collapse: bounding
 * logit spectral norm bounds attention entropy from below).
 *
 * Power iteration estimates sigma(W) (largest singular value); the
 * normalized weight W/sigma(W) is 1-Lipschitz. We provide:
 *   - wubu_sn_estimate: power-iteration spectral norm estimate
 *   - wubu_sn_normalize: divide W in place to cap at sigma_max
 *
 * For the WuBu transformer this is the missing stability piece: apply
 * after every RAdam step on the gyrolinear weights.
 */
#include "wubu_specnorm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

float wubu_sn_estimate(const float* W,int rows,int cols,
                        int iters,unsigned* seed){
    /* v <- W^T W v / |W^T W v|, sigma = |Wv| */
    float* v=malloc(sizeof(float)*(size_t)cols);
    if(!v)return 0;
    float vn2=0;
    for(int j=0;j<cols;j++){
        *seed=*seed*1103515245u+12345u;
        v[j]=((float)((*seed>>16)%2000))/1000.0f-1.0f;
        vn2+=v[j]*v[j];
    }
    float vn=sqrtf(vn2);
    if(vn<1e-12f){free(v);return 0;}
    for(int j=0;j<cols;j++)v[j]/=vn;
    float sigma=0;
    for(int it=0;it<iters;it++){
        /* u = Wv; sigma estimate = |u| / |v| with |v|=1 */
        float u[512];
        int rc=rows<512?rows:512;
        float un2=0;
        for(int i=0;i<rc;i++){
            float acc=0;
            int cc=cols<512?cols:512;
            for(int j=0;j<cc;j++)acc+=W[(size_t)i*cols+j]*v[j];
            u[i]=acc;un2+=acc*acc;
        }
        sigma=sqrtf(un2);
        /* v' = W^T u */
        float n2=0;
        int cc=cols<512?cols:512;
        memset(v,0,sizeof(float)*(size_t)cc);
        for(int i=0;i<rc;i++)
            for(int j=0;j<cc;j++)v[j]+=W[(size_t)i*cols+j]*u[i];
        for(int j=0;j<cc;j++)n2+=v[j]*v[j];
        float nv=sqrtf(n2);
        if(nv<1e-12f){sigma=0;goto done;}
        for(int j=0;j<cc;j++)v[j]/=nv;
    }
done:
    free(v);
    return sigma;
}

void wubu_sn_normalize(float* W,int rows,int cols,float sigma_max,
                        int iters,unsigned* seed){
    float s=wubu_sn_estimate(W,rows,cols,iters,seed);
    if(s>sigma_max&&s>1e-10f){
        float f=sigma_max/s;
        int n=rows*cols;
        for(int i=0;i<n;i++)W[i]*=f;
    }
}
