/* test_wubu_hmix.c -- GAP-D029 gates
 *  G1 product dist >= each factor alone
 *  G2 zero spherical diff -> product == hyperbolic
 *  G3 project: hyperbolic in ball, spherical on unit sphere
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hmix.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== H×S Product Manifold Tests ===\n\n");
    const int Dh=4,Ds=4,D=Dh+Ds;
    float c=1.0f;

    float x[D],y[D];
    unsigned rs=42u;
    for(int i=0;i<D;i++){
        rs=rs*1103515245u+12345u;
        x[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    /* spherical part: normalize to unit sphere */
    {
        float sn2=0;
        for(int d=Dh;d<D;d++)sn2+=x[d]*x[d];
        float sn=sqrtf(sn2);
        if(sn>1e-10f)for(int d=Dh;d<D;d++)x[d]/=sn;
    }

    printf("  g1_product_geq_factors...");
    {
        float dp=wubu_hm_product_dist(x,y,Dh,Ds,c);
        float dh=wubu_hm_hyper_dist(x,y,Dh,c);
        float dot=0;
        for(int d=Dh;d<D;d++)dot+=x[d]*y[d];
        if(dot>1)dot=1;if(dot<-1)dot=-1;
        float ds=acosf(dot);
        CHECK(dp>=dh-1e-5f);
        CHECK(dp>=ds-1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_zero_sphere_eq_hyper...");
    {
        float x2[8],y2[8];
        memcpy(x2,x,sizeof(float)*D);
        memcpy(y2,y,sizeof(float)*D);
        for(int d=Dh;d<D;d++)y2[d]=x[d];
        float dp=wubu_hm_product_dist(x2,y2,Dh,Ds,c);
        float dh=wubu_hm_hyper_dist(x2,y2,Dh,c);
        CHECK(fabsf(dp-dh)<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g3_project...");
    {
        float p[D];
        for(int i=0;i<D;i++)p[i]=((i%3)==0)?5.0f:0.01f*(i+1);
        wubu_hm_project(p,Dh,Ds,c);
        /* hyperbolic norm < 1 */
        float n2=0;for(int d=0;d<Dh;d++)n2+=p[d]*p[d];
        CHECK(n2<=1.0f+1e-5f);
        /* spherical norm = 1 */
        float sn2=0;
        for(int d=Dh;d<D;d++)sn2+=p[d]*p[d];
        CHECK(fabsf(sqrtf(sn2)-1.0f)<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
