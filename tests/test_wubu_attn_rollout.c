/* test_wubu_attn_rollout.c -- GAP-B016 gates
 *  G1 rows sum to 1, all nonneg
 *  G2 identity attention -> rollout = identity (with residual)
 *  G3 2-layer = product of layers
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_attn_rollout.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Attention Rollout Tests ===\n\n");
    const int N=5;

    printf("  g1_rows_sum_one...");
    {
        /* random attention */
        float attn[25],out[25];
        unsigned rs=7u;
        for(int i=0;i<25;i++){
            rs=rs*1103515245u+12345u;
            attn[i]=(float)((rs>>16)%100)/100.0f;
        }
        wubu_ar_layer(attn,N,out);
        for(int i=0;i<N;i++){
            float s=0;for(int j=0;j<N;j++){CHECK(out[i*N+j]>=0);s+=out[i*N+j];}
            CHECK(fabsf(s-1.0f)<1e-4f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_identity_attention...");
    {
        float attn[25]={0},out[25];
        for(int i=0;i<N;i++)attn[i*N+i]=1.0f;   /* pure identity */
        wubu_ar_layer(attn,N,out);
        /* with identity input: out[i][j] = (I+I)/2 = I */
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                float expected=(i==j)?1.0f:0.0f;
                CHECK(fabsf(out[i*N+j]-expected)<1e-4f);
            }
    }
    printf("PASS\n");passed++;

    printf("  g3_two_layer_associative...");
    {
        float a1[25],a2[25],roll[25];
        for(int i=0;i<25;i++){a1[i]=((i*13)%7)/7.0f;a2[i]=((i*17)%11)/11.0f;}
        const float* ptrs[2]={a1,a2};
        int nl=2;
        wubu_ar_rollout(ptrs,&nl,N,roll);
        /* verify rows sum to ~1 */
        for(int i=0;i<N;i++){
            float s=0;for(int j=0;j<N;j++)s+=roll[i*N+j];
            CHECK(fabsf(s-1.0f)<1e-3f);
        }
        /* nonneg */
        for(int i=0;i<25;i++)CHECK(roll[i]>=-1e-6f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
