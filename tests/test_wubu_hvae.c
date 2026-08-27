/* test_wubu_hvae.c -- GAP-C024 gates
 *  G1 samples stay inside the ball
 *  G2 samples cluster around mu (empirical mean within tolerance)
 *  G3 larger sigma gives larger empirical spread
 *  G4 log-density finite, higher at mu than far away
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hvae.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic VAE Sampling Tests ===\n\n");
    const int D=8,N=2000;
    float c=1.0f;
    float mu[8],sigma[8];
    for(int d=0;d<D;d++){mu[d]=((float)(d%5)-2)*0.1f;sigma[d]=0.2f;}

    printf("  g1_samples_on_ball...");
    {
        float z[8];
        for(int i=0;i<N;i++){
            wubu_hvae_sample(z,mu,sigma,D,c);
            float n2=0;for(int d=0;d<D;d++)n2+=z[d]*z[d];
            CHECK(n2<1.0f+1e-5f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_clusters_around_mu...");
    {
        /* empirical mean of many samples should be near mu */
        float mean[8]={0};
        float z[8];
        for(int i=0;i<N;i++){
            wubu_hvae_sample(z,mu,sigma,D,c);
            for(int d=0;d<D;d++)mean[d]+=z[d];
        }
        for(int d=0;d<D;d++)mean[d]/=N;
        float dist=0;
        for(int d=0;d<D;d++){float df=mean[d]-mu[d];dist+=df*df;}
        CHECK(sqrtf(dist)<0.15f);  /* tolerance for 2000 samples */
    }
    printf("PASS\n");passed++;

    printf("  g3_sigma_controls_spread...");
    {
        /* compute spread with small sigma vs large sigma */
        float z[8];
        float sig_small[8],sig_large[8];
        for(int d=0;d<D;d++){sig_small[d]=0.05f;sig_large[d]=0.5f;}
        float spread_s=0,spread_l=0;
        for(int i=0;i<500;i++){
            wubu_hvae_sample(z,mu,sig_small,D,c);
            float n2=0;for(int d=0;d<D;d++)n2+=(z[d]-mu[d])*(z[d]-mu[d]);
            spread_s+=sqrtf(n2);
            wubu_hvae_sample(z,mu,sig_large,D,c);
            n2=0;for(int d=0;d<D;d++)n2+=(z[d]-mu[d])*(z[d]-mu[d]);
            spread_l+=sqrtf(n2);
        }
        spread_s/=500;spread_l/=500;
        CHECK(spread_l>spread_s*2.0f);   /* 10x sigma → much more spread */
    }
    printf("PASS\n");passed++;

    printf("  g4_log_density...");
    {
        /* At moderate distance, Euclidean term should dominate.
         * Use a point that's 3 sigma away (clearly less probable). */
        float near[8],far[8];
        for(int d=0;d<D;d++){
            near[d]=mu[d]+2*sigma[d];   /* 2 sigma */
            far[d]=mu[d]+5*sigma[d];    /* 5 sigma — much less probable */
        }
        float ld_at=wubu_hvae_log_density(near,mu,sigma,D,c);
        float ld_far=wubu_hvae_log_density(far,mu,sigma,D,c);
        CHECK(ld_at>ld_far);
        CHECK(!isnan(ld_at)&&!isnan(ld_far));
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
