/* test_wubu_specnorm.c -- GAP-C043 gates
 *  G1 known matrix: diag(3,2) → sigma estimate ≈ 3
 *  G2 normalize caps sigma: post-normalization estimate <= sigma_max*1.05
 *  G3 small-norm matrix untouched (no unnecessary scaling)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_specnorm.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Spectral Normalization Tests ===\n\n");

    printf("  g1_known_sigma...");
    {
        /* diagonal 4x4 with entries 3,2,1,0.5 → sigma=3 */
        float W[16]={0};
        W[0]=3;W[5]=2;W[10]=1;W[15]=0.5f;
        unsigned seed=42u;
        float s=wubu_sn_estimate(W,4,4,30,&seed);
        printf("[est=%.4f want=3] ",(double)s);
        CHECK(fabsf(s-3.0f)<0.05f);
    }
    printf("PASS\n");passed++;

    printf("  g2_normalize_caps...");
    {
        /* random 8x8 scaled large */
        float W[64];
        unsigned rs=7u;
        for(int i=0;i<64;i++){
            rs=rs*1103515245u+12345u;
            W[i]=((float)((rs>>16)%2000))/1000.0f-1.0f;
        }
        unsigned seed=99u;
        wubu_sn_normalize(W,8,8,1.0f,40,&seed);
        seed=100u;
        float s=wubu_sn_estimate(W,8,8,40,&seed);
        printf("[post=%.4f cap=1] ",(double)s);
        CHECK(s<=1.05f);
    }
    printf("PASS\n");passed++;

    printf("  g3_small_untouched...");
    {
        float W[4]={0.01f,0,0,0.02f};
        float before[4];memcpy(before,W,sizeof(W));
        unsigned seed=1u;
        wubu_sn_normalize(W,2,2,10.0f,20,&seed);
        CHECK(fabsf(W[0]-before[0])<1e-9f);
        CHECK(fabsf(W[3]-before[3])<1e-9f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
