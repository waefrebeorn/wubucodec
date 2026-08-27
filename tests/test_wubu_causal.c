/* test_wubu_causal.c -- GAP-C035 gates
 *  G1 upper triangle zeroed after apply
 *  G2 rows renormalized (sum over allowed = 1)
 *  G3 respected() detects violations before, passes after
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_causal.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Causal Mask Tests ===\n\n");
    const int N=6;

    printf("  g3_detect_then_pass...");
    {
        float attn[36];
        unsigned rs=42u;
        for(int i=0;i<36;i++){
            rs=rs*1103515245u+12345u;
            attn[i]=(float)((rs>>16)%1000)/1000.0f;
        }
        /* full attention violates causality */
        CHECK(!wubu_causal_respected(attn,N,1e-9f));
        /* apply mask */
        wubu_causal_apply(attn,N);
        CHECK(wubu_causal_respected(attn,N,1e-9f));
    }
    printf("PASS\n");passed++;

    printf("  g1_upper_zeroed...");
    {
        float attn[36];
        for(int i=0;i<36;i++)attn[i]=1.0f;   /* all-ones: worst case */
        wubu_causal_apply(attn,N);
        for(int i=0;i<N;i++)
            for(int j=i+1;j<N;j++)
                CHECK(attn[i*N+j]==0.0f);
        /* diagonal + lower strictly positive */
        for(int i=0;i<N;i++)CHECK(attn[i*N+i]>0.0f);
    }
    printf("PASS\n");passed++;

    printf("  g2_rows_renormalized...");
    {
        float attn[36]={0};
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                attn[i*N+j]=(float)((i*7+j*13)%11)/11.0f;
        wubu_causal_apply(attn,N);
        for(int i=0;i<N;i++){
            float s=0;
            for(int j=0;j<=i;j++)s+=attn[i*N+j];
            CHECK(fabsf(s-1.0f)<1e-5f);
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
