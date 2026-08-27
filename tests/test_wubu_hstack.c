/* test_wubu_hstack.c -- GAP-C040 gates
 *  G1 4-layer stack: outputs on-ball, finite
 *  G2 ANTI-OVERSMOOTHING: node distinctness does NOT collapse
 *     (mean pairwise distance after stack > 10% of input distinctness)
 *  G3 alpha=0 → output == input exactly (residual identity)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hstack.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic GCN Stack Tests ===\n\n");
    const int N=6,D=8,L=4;
    float c=1.0f;

    /* ring graph */
    int adj_idx[12]={1,5, 0,2, 1,3, 2,4, 3,5, 4,0};
    int adj_ptr[7]={0,2,4,6,8,10,12};

    float x[N*D];
    unsigned rs=42u;
    for(int i=0;i<N*D;i++){
        rs=rs*1103515245u+12345u;
        x[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=x[i*D+d]*x[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)x[i*D+d]*=s;}
    }

    /* per-layer weights: small random (near-identity behavior) */
    float W[L*D*D];
    for(int i=0;i<L*D*D;i++){
        rs=rs*1103515245u+12345u;
        W[i]=(float)((rs>>16)%1000)/50000.0f-0.01f;
    }

    printf("  g3_alpha0_identity...");
    {
        float out[N*D];
        CHECK(wubu_hstack_forward(x,adj_idx,adj_ptr,NULL,W,L,N,D,c,0.0f,out)==0);
        for(int i=0;i<N*D;i++)
            CHECK(fabsf(out[i]-x[i])<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g1_on_ball_4layers...");
    {
        float out[N*D];
        CHECK(wubu_hstack_forward(x,adj_idx,adj_ptr,NULL,W,L,N,D,c,0.3f,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=out[i*D+d]*out[i*D+d];
            CHECK(n2<1.0f);
            for(int d=0;d<D;d++)CHECK(!isnan(out[i*D+d]));
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_no_oversmoothing...");
    {
        float out[N*D];
        wubu_hstack_forward(x,adj_idx,adj_ptr,NULL,W,L,N,D,c,0.3f,out);
        float in_dist=wubu_hstack_distinctness(x,N,D,c);
        float out_dist=wubu_hstack_distinctness(out,N,D,c);
        printf("[in=%.4f out=%.4f] ",(double)in_dist,(double)out_dist);
        /* the anti-oversmoothing invariant: nodes stay distinguishable */
        CHECK(out_dist>in_dist*0.1f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
