/* test_wubu_hgat.c -- GAP-C041 gates
 *  G1 outputs on-ball, finite
 *  G2 closer neighbors get MORE weight (attention is distance-ordered):
 *     a node with a very close neighbor and a far one → output pulled
 *     toward the close one (output closer to near neighbor than far)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hgat.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float gd(const float*a,const float*b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=a[d]-b[d];ab2+=df*df;a2+=a[d]*a[d];b2+=b[d]*b[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== Hyperbolic GAT Tests ===\n\n");
    const int N=3,D=8;
    float c=1.0f;

    /* node 0 center; node 1 VERY close to 0; node 2 FAR from 0 */
    float x[N*D];
    for(int d=0;d<D;d++){
        x[0*D+d]=((float)(d%5)-2)*0.02f;
        x[1*D+d]=x[d]+0.03f;              /* near */
        x[2*D+d]=((float)(d%4)-1.5f)*0.35f; /* far */
    }
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=x[i*D+d]*x[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)x[i*D+d]*=s;}
    }
    /* star: 0 connected to 1 and 2 */
    int adj_idx[2]={1,2};
    int adj_ptr[7]={0,2,2,2,2,2,2};

    printf("  g1_on_ball_finite...");
    {
        float out[N*D];
        CHECK(wubu_hgat_forward(x,adj_idx,adj_ptr,NULL,NULL,N,D,c,0.5f,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=out[i*D+d]*out[i*D+d];
            CHECK(n2<1.0f);
            for(int d=0;d<D;d++)CHECK(!isnan(out[i*D+d]));
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_close_neighbor_dominates...");
    {
        float out[N*D];
        wubu_hgat_forward(x,adj_idx,adj_ptr,NULL,NULL,N,D,c,0.5f,out);
        /* output of node 0 should be closer to node 1 than to node 2
         * (node 1 was nearer pre-aggregation) */
        float d_to_near=gd(out,x+D,D,c);
        float d_to_far=gd(out,x+2*D,D,c);
        CHECK(d_to_near<d_to_far);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
