/* test_wubu_multihop.c -- GAP-B018 gates
 *  G1 outputs on-ball, finite
 *  G2 1-hop-only (tau small, max_hops=1) ≈ self-dominated output
 *  G3 multi-hop pulls node toward its 2-hop cluster (info propagates)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_multihop.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Multi-hop Aggregation Tests ===\n\n");
    /* path graph: 0-1-2-3-4 (line), node 0 queried */
    const int N=5,D=4;
    float c=1.0f;
    int adj_idx[8]={1, 0,2, 1,3, 2,4, 3};
    int adj_ptr[6]={0,1,3,5,7,8};

    /* node features: node i sits at increasing radius along e0 */
    float x[N*D];
    memset(x,0,sizeof(x));
    for(int i=0;i<N;i++)x[i*D+0]=0.1f*(i+1);

    printf("  g1_on_ball_finite...");
    {
        float out[N*D];
        CHECK(wubu_mh_aggregate(x,adj_idx,adj_ptr,N,D,c,1.0f,2,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++){CHECK(!isnan(out[i*D+d]));n2+=out[i*D+d]*out[i*D+d];}
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_self_dominant_short_tau...");
    {
        /* tau -> tiny: neighbors contribute ~nothing; out ≈ exp0(log0(self)) = self */
        float out[N*D];
        wubu_mh_aggregate(x,adj_idx,adj_ptr,N,D,c,0.01f,1,out);
        for(int d=0;d<D;d++)
            CHECK(fabsf(out[d]-x[d])<0.05f);
    }
    printf("PASS\n");passed++;

    printf("  g3_multihop_pulls...");
    {
        /* with long tau and 2 hops, node 0's aggregate should move TOWARD
         * the mean of nodes 1-2 (positive e0 direction) vs its own value */
        float out[N*D];
        wubu_mh_aggregate(x,adj_idx,adj_ptr,N,D,c,5.0f,2,out);
        /* x[0]=0.1; neighbors at 0.2,0.3 pull it up */
        CHECK(out[0]>x[0]);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
