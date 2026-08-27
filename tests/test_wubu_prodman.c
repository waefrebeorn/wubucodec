/* test_wubu_prodman.c -- GAP-D014 gates
 *  G1 product distance >= each factor distance alone
 *  G2 zero euclidean diff → product dist == hyperbolic dist
 *  G3 symmetric
 *  G4 project keeps hyperbolic part on-ball, leaves euclidean part
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_prodman.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Product Manifold Tests ===\n\n");
    const int D_hyp=4,D_euc=4,D=D_hyp+D_euc;
    float c=1.0f;

    float x[D],y[D];
    for(int d=0;d<D;d++){
        x[d]=((float)((d*37)%89)/89.0f-0.5f)*0.5f;
        y[d]=((float)((d*53)%97)/97.0f-0.5f)*0.5f;
    }
    /* ensure hyperbolic parts are inside ball */
    wubu_pm_project(x,D_hyp,D_euc,c);
    wubu_pm_project(y,D_hyp,D_euc,c);

    printf("  g1_product_geq_factors...");
    {
        float dp=wubu_pm_product_dist(x,y,D_hyp,D_euc,c);
        float dh=wubu_pm_hyper_dist(x,y,D_hyp,c);
        CHECK(dp>=dh);
    }
    printf("PASS\n");passed++;

    printf("  g2_zero_euc_eq_hyp...");
    {
        float x2[D],y2[D];
        memcpy(x2,x,sizeof(float)*D);
        memcpy(y2,y,sizeof(float)*D);
        /* make euclidean parts identical */
        for(int d=D_hyp;d<D;d++)y2[d]=x[d];
        float dp=wubu_pm_product_dist(x2,y2,D_hyp,D_euc,c);
        float dh=wubu_pm_hyper_dist(x2,y2,D_hyp,c);
        CHECK(fabsf(dp-dh)<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g3_symmetric...");
    {
        float dxy=wubu_pm_product_dist(x,y,D_hyp,D_euc,c);
        float dyx=wubu_pm_product_dist(y,x,D_hyp,D_euc,c);
        CHECK(fabsf(dxy-dyx)<1e-6f);
    }
    printf("PASS\n");passed++;

    printf("  g4_project...");
    {
        float p[D];
        for(int d=0;d<D;d++)p[d]=(d<4)?10.0f:3.0f;  /* way off-ball */
        wubu_pm_project(p,D_hyp,D_euc,c);
        /* hyperbolic clamped */
        float n2=0;for(int d=0;d<D_hyp;d++)n2+=p[d]*p[d];
        CHECK(n2<1.0f+1e-5f);
        /* euclidean untouched */
        CHECK(fabsf(p[4]-3.0f)<1e-6f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
