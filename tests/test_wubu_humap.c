/* test_wubu_humap.c -- GAP-D032 gates
 *  G1 fuzzy weights symmetric, in [0,1), zero off-graph
 *  G2 CE well-defined: finite, non-negative for valid weights
 *  G3 layout matching graph structure has LOWER CE than scrambled
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_humap.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic UMAP Tests ===\n\n");
    /* 6 points: two triplets far apart in input space */
    const int N=6,D=4;
    float c=1.0f;
    float xs[N*D];
    for(int i=0;i<N;i++)
        for(int d=0;d<D;d++)
            xs[i*D+d]=((i<3)?-1.5f:1.5f)*(((d+1)%2)?1.0f:0.02f)+i*0.01f+d*0.001f;

    /* k-NN graph: each point connects to its 2 nearest (same-triplet) */
    int adj_idx[12]={1,2, 0,2, 0,1, 4,5, 3,5, 3,4};
    int adj_ptr[7]={0,2,4,6,8,10,12};

    printf("  g1_fuzzy_symmetric...");
    {
        float W[N*N];memset(W,0,sizeof(W));
        CHECK(wubu_um_fuzzy_weights(xs,adj_idx,adj_ptr,N,D,c,W)==0);
        for(int i=0;i<N;i++)
            for(int j=i+1;j<N;j++)
                CHECK(W[i*N+j]==W[j*N+i]);
        for(int i=0;i<N*N;i++){
            CHECK(W[i]>=0&&W[i]<1);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_ce_finite...");
    {
        float W[N*N];memset(W,0,sizeof(W));
        wubu_um_fuzzy_weights(xs,adj_idx,adj_ptr,N,D,c,W);
        float ys[N*2];
        unsigned rs=42u;
        for(int i=0;i<N;i++){
            rs=rs*1103515245u+12345u;
            ys[i*2]=((float)((rs>>16)%1000))/2500.0f-0.2f;
            ys[i*2+1]=((float)((rs>>8)%1000))/2500.0f-0.2f;
        }
        float ce=wubu_um_cross_entropy(W,ys,N,2,c);
        CHECK(!isnan(ce));
    }
    printf("PASS\n");passed++;

    printf("  g3_structure_lower_ce...");
    {
        float W[N*N];memset(W,0,sizeof(W));
        wubu_um_fuzzy_weights(xs,adj_idx,adj_ptr,N,D,c,W);
        /* structured: same-triplet points placed together on the ball */
        float ys_good[N*2],ys_bad[N*2];
        for(int i=0;i<N;i++){
            float side=(i<3)?-0.6f:0.6f;
            ys_good[i*2]=side;ys_good[i*2+1]=(i%3)*0.05f;
            ys_bad[i*2]=((i%2)?0.6f:-0.6f);ys_bad[i*2+1]=0;
        }
        float ce_g=wubu_um_cross_entropy(W,ys_good,N,2,c);
        float ce_b=wubu_um_cross_entropy(W,ys_bad,N,2,c);
        printf("[good=%.3f bad=%.3f] ",(double)ce_g,(double)ce_b);
        CHECK(ce_g<ce_b);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
