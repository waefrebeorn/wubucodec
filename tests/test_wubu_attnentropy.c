/* test_wubu_attnentropy.c -- GAP-B017 gates
 *  G1 uniform attention → max entropy log(N)
 *  G2 one-hot attention → zero entropy (the collapse extreme)
 *  G3 collapse flag: one-hot flagged, uniform not flagged
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_attnentropy.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Attention Entropy Monitor Tests ===\n\n");
    const int N=8,H=2;

    printf("  g1_uniform_max_entropy...");
    {
        float attn[H*N*N];
        for(int h=0;h<H;h++)
            for(int i=0;i<N;i++)
                for(int j=0;j<N;j++)
                    attn[(h*N+i)*N+j]=1.0f/N;
        float Hm=wubu_ae_mean_entropy(attn,N,H);
        float logN=logf((float)N);
        printf("[H=%.4f logN=%.4f] ",(double)Hm,(double)logN);
        CHECK(fabsf(Hm-logN)<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("  g2_onehot_zero_entropy...");
    {
        float attn[N*N];
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                attn[i*N+j]=(i==j)?1.0f:0.0f;
        for(int i=0;i<N;i++){
            float h=wubu_ae_row_entropy(attn+i*N,N);
            CHECK(h<1e-6f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_collapse_flag...");
    {
        /* one-hot block: collapsed */
        float onehot[N*N];
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)onehot[i*N+j]=(i==j)?1.0f:0.0f;
        CHECK(wubu_ae_collapsed(onehot,N,1,0.1f*logf((float)N))==1);
        /* uniform block: healthy */
        float uni[N*N];
        for(int i=0;i<N*N;i++)uni[i]=1.0f/N;
        CHECK(wubu_ae_collapsed(uni,N,1,0.1f*logf((float)N))==0);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
