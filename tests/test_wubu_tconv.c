/* test_wubu_tconv.c -- GAP-C048 gates
 *  G1 outputs on-ball for the full sequence
 *  G2 constant sequence → constant output (translation invariance)
 *  G3 smoothing kernel reduces frame-to-frame jitter vs raw
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_tconv.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Temporal Conv Tests ===\n\n");
    const int T=12,D=8,K=3;
    float c=1.0f;

    /* noisy sequence around a base direction */
    float seq[T*D];
    unsigned rs=42u;
    float base[8];
    for(int d=0;d<D;d++)base[d]=((float)(d%5)-2)*0.08f;
    for(int t=0;t<T;t++)
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/4000.0f-0.0125f;
            seq[t*D+d]=base[d]+noise;
        }
    /* project all into ball */
    for(int t=0;t<T;t++){
        float n2=0;for(int d=0;d<D;d++)n2+=seq[t*D+d]*seq[t*D+d];
        if(n2>0.9f){float s=sqrtf(0.9f/n2);for(int d=0;d<D;d++)seq[t*D+d]*=s;}
    }

    printf("  g1_on_ball...");
    {
        float out[T*D];
        CHECK(wubu_tc_conv1d(seq,T,D,K,c,NULL,0.0f,7u,out)==0);
        for(int t=0;t<T;t++){
            float n2=0;
            for(int d=0;d<D;d++){CHECK(!isnan(out[t*D+d]));n2+=out[t*D+d]*out[t*D+d];}
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_constant_invariant...");
    {
        const int T2=6;
        float cst[T2*D],out[T2*D];
        for(int t=0;t<T2;t++)
            for(int d=0;d<D;d++)cst[t*D+d]=base[d]*1.5f;
        CHECK(wubu_tc_conv1d(cst,T2,D,K,c,NULL,0.0f,7u,out)==0);
        /* all outputs identical to each other */
        for(int t=1;t<T2;t++)
            for(int d=0;d<D;d++)
                CHECK(fabsf(out[t*D+d]-out[d])<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("  g3_smoothing_reduces_jitter...");
    {
        float kernel[3]={0.25f,0.5f,0.25f};
        float out[T*D];
        CHECK(wubu_tc_conv1d(seq,T,D,K,c,kernel,0.0f,7u,out)==0);
        /* mean |delta| between consecutive frames: conv < raw */
        double jit_raw=0,jit_conv=0;
        for(int t=1;t<T;t++)
            for(int d=0;d<D;d++){
                jit_raw+=fabs(seq[t*D+d]-seq[(t-1)*D+d]);
                jit_conv+=fabs(out[t*D+d]-out[(t-1)*D+d]);
            }
        jit_raw/=(double)(T-1)*D;jit_conv/=(double)(T-1)*D;
        printf("[raw=%.4f conv=%.4f] ",(double)jit_raw,(double)jit_conv);
        CHECK(jit_conv<jit_raw);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
