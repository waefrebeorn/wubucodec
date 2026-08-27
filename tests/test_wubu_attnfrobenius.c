/* test_wubu_attnfrobenius.c -- GAP-B019 gates
 *  G1 uniform attention: ||P||_F ≈ sqrt(n_heads) (each row uniform)
 *     and uniformity ≈ 1
 *  G2 one-hot attention: ||P||_F ≈ sqrt(n_heads*N), uniformity ≈ 0
 *  G3 monotonicity: interpolating uniform->one-hot decreases uniformity
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_attnfrobenius.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Attention Frobenius Monitor Tests ===\n\n");
    const int N=8,H=2;

    printf("  g1_uniform...");
    {
        float attn[H*N*N];
        for(int i=0;i<H*N*N;i++)attn[i]=1.0f/N;
        float f=wubu_af_frobenius(attn,N,H);
        float u=wubu_af_uniformity(attn,N,H);
        printf("[F=%.4f (want %.4f) U=%.3f] ",
               (double)f,(double)sqrtf((float)H),(double)u);
        CHECK(fabsf(f-sqrtf((float)H))<1e-3f);
        CHECK(u>0.99f);
    }
    printf("PASS\n");passed++;

    printf("  g2_onehot...");
    {
        float attn[H*N*N];
        for(int h=0;h<H;h++)
            for(int i=0;i<N;i++)
                for(int j=0;j<N;j++)
                    attn[(h*N+i)*N+j]=(i==j)?1.0f:0.0f;
        float f=wubu_af_frobenius(attn,N,H);
        float u=wubu_af_uniformity(attn,N,H);
        printf("[F=%.4f (want %.4f) U=%.3f] ",
               (double)f,(double)sqrtf((float)(H*N)),(double)u);
        CHECK(fabsf(f-sqrtf((float)(H*N)))<1e-3f);
        CHECK(u<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g3_monotone_interpolation...");
    {
        /* blend factor a: 0=uniform, 1=one-hot */
        float prev_u=1.1f;
        int mono=1;
        for(int step=0;step<=10;step++){
            float a=step/10.0f;
            float attn[N*N];
            for(int i=0;i<N;i++)
                for(int j=0;j<N;j++){
                    float onehot=(i==j)?1.0f:0.0f;
                    float uni=1.0f/N;
                    attn[i*N+j]=a*onehot+(1-a)*uni;
                }
            float u=wubu_af_uniformity(attn,N,1);
            if(u>prev_u+1e-4f){mono=0;}
            prev_u=u;
        }
        CHECK(mono);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
