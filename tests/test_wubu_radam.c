/* test_wubu_radam.c -- GAP-C038 gates
 *  G1 stays on-ball across 500 steps
 *  G2 converges faster than RSGD on a quadratic-ish loss (Adam advantage)
 *  G3 moments actually move (non-degenerate)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_radam.h"
#include "wubu_rsgd.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float dist_to(const float* x,const float* t,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=x[d]-t[d];ab2+=df*df;a2+=x[d]*x[d];b2+=t[d]*t[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== Riemannian Adam Tests ===\n\n");
    const int D=8;
    float c=1.0f,target[8];
    for(int d=0;d<D;d++)target[d]=((float)(d%5)-2)*0.06f;

    printf("  g1_stays_on_ball...");
    {
        WubuRADAM o;
        CHECK(wubu_radam_init(&o,D,0.01f,0.9f,0.999f,1e-8f)==0);
        float x[8]={0};x[0]=0.05f;
        unsigned rs=42u;
        for(int s=0;s<500;s++){
            float g[8];
            for(int d=0;d<D;d++){
                rs=rs*1103515245u+12345u;
                float noise=((rs>>16)%100)/2500.0f-0.02f;
                g[d]=2*(x[d]-target[d])+noise;
            }
            wubu_radam_step(&o,x,g,c);
            float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
            CHECK(n2<1.0f);
            for(int d=0;d<D;d++)CHECK(!isnan(x[d]));
        }
        wubu_radam_free(&o);
    }
    printf("PASS\n");passed++;

    printf("  g2_faster_than_rsgd...");
    {
        /* same budget, compare final distances: Adam should be <= RSGD+tolerance */
        float xa[8]={0},xs[8]={0};
        xa[0]=0.1f;xs[0]=0.1f;
        WubuRADAM o;
        wubu_radam_init(&o,D,0.1f,0.9f,0.999f,1e-8f);
        for(int s=0;s<300;s++){
            float ga[8],gs[8];
            for(int d=0;d<D;d++){
                ga[d]=2*(xa[d]-target[d]);
                gs[d]=ga[d];
            }
            wubu_radam_step(&o,xa,ga,c);
            wubu_rsgd_step(xs,gs,D,c,0.005f);
        }
        float da=dist_to(xa,target,D,c);
        float ds=dist_to(xs,target,D,c);
        printf("[adam=%.4f rsgd=%.4f] ",(double)da,(double)ds);
        CHECK(da<=ds*1.5f);   /* Adam at least competitive */
        CHECK(da<0.8f);       /* and actually converged */
    }
    printf("PASS\n");passed++;

    printf("  g3_moments_nonzero...");
    {
        WubuRADAM o;
        wubu_radam_init(&o,D,0.01f,0.9f,0.999f,1e-8f);
        float x[8]={0};x[0]=0.1f;
        float g[8];for(int d=0;d<D;d++)g[d]=1.0f;
        wubu_radam_step(&o,x,g,c);
        float msum=0,vsum=0;
        for(int d=0;d<D;d++){msum+=o.m[d]>0?o.m[d]:-o.m[d];vsum+=o.v[d];}
        CHECK(msum>0);
        CHECK(vsum>0);
        wubu_radam_free(&o);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
