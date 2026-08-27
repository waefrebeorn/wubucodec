/* test_wubu_hope.c -- GAP-C021 gates: hyperbolic rotary positional encoding
 *  G1 monotone: norm grows with position (positions distinguishable)
 *  G2 finite at long positions (10000 tokens)
 *  G3 different positions give different encodings of the same vector
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hope.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

static float l2norm(const float* v,int D){
    float n=0;for(int d=0;d<D;d++)n+=v[d]*v[d];
    return sqrtf(n);
}
int main(void){
    printf("=== HoPE Positional Encoding Tests ===\n\n");
    const int D=64;

    float x[D];
    for(int d=0;d<D;d++)x[d]=((float)(d%7)-3.0f)*0.1f;

    printf("  g1_monotone_norm...");
    {
        float prev=-1.0f;
        for(int pos=1;pos<=20;pos++){
            float out[D];
            wubu_hope_encode(out,x,D,pos,10000.0f);
            float n=l2norm(out,D);
            CHECK(n>prev-1e-6f);   /* non-decreasing */
            CHECK(!isnan(n));
            prev=n;
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_long_position_finite...");
    {
        for(int pos=100;pos<=10000;pos*=10){
            float out[D];
            wubu_hope_encode(out,x,D,pos,10000.0f);
            for(int d=0;d<D;d++)CHECK(!isnan(out[d])&&!isinf(out[d]));
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_positions_distinguishable...");
    {
        float e1[64],e2[64],e3[64];
        wubu_hope_encode(e1,x,D,5,10000.0f);
        wubu_hope_encode(e2,x,D,50,10000.0f);
        wubu_hope_encode(e3,x,D,500,10000.0f);
        /* pairwise L2 distances all > 0 */
        float d12=0,d23=0,d13=0;
        for(int d=0;d<D;d++){d12+=(e1[d]-e2[d])*(e1[d]-e2[d]);
                            d23+=(e2[d]-e3[d])*(e2[d]-e3[d]);
                            d13+=(e1[d]-e3[d])*(e1[d]-e3[d]);}
        CHECK(sqrtf(d12)>1e-4f);
        CHECK(sqrtf(d23)>1e-4f);
        CHECK(sqrtf(d13)>sqrtf(d12));  /* farther positions differ more */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
