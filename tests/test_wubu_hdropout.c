/* test_wubu_hdropout.c -- GAP-C045 gates
 *  G1 output stays on-ball for many random inputs and rates
 *  G2 rate=0 → identity (exact round trip through tangent space)
 *  G3 dropout actually perturbs (output != input at high rate, on average)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hdropout.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Dropout Tests ===\n\n");
    const int D=16;
    float c=1.0f;

    printf("  g1_on_ball...");
    {
        unsigned seed=42u;
        for(int trial=0;trial<200;trial++){
            float x[D],out[D];
            for(int d=0;d<D;d++){
                seed++;
                x[d]=((float)((seed>>8)%2000))/10000.0f-0.1f;
            }
            float rate=0.1f+0.6f*(trial%6)/5.0f;
            CHECK(wubu_hd_apply(x,D,c,rate,&seed,out)==0);
            float n2=0;
            for(int d=0;d<D;d++){CHECK(!isnan(out[d]));n2+=out[d]*out[d];}
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_rate0_identity...");
    {
        float x[D],out[D];
        unsigned seed=7u;
        for(int d=0;d<D;d++)x[d]=((float)(d%5)-2)*0.06f;
        CHECK(wubu_hd_apply(x,D,c,0.0f,&seed,out)==0);
        for(int d=0;d<D;d++)CHECK(fabsf(out[d]-x[d])<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("  g3_dropout_perturbs...");
    {
        float x[D],out[D];
        unsigned seed=99u;
        for(int d=0;d<D;d++)x[d]=((float)(d%7)-3)*0.05f;
        double diff_sum=0;
        for(int t=0;t<50;t++){
            wubu_hd_apply(x,D,c,0.9f,&seed,out);
            float df=0;
            for(int d=0;d<D;d++){float dd=out[d]-x[d];df+=dd*dd;}
            diff_sum+=sqrtf(df);
        }
        float mean_diff=(float)(diff_sum/50);
        CHECK(mean_diff>1e-4f);   /* noise actually applied */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
