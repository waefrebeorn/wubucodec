/* test_wubu_hdpc.c -- GAP-D021 gates
 *  G1 finds >=2 cluster centers for two separated blobs
 *  G2 every point assigned
 *  G3 points assigned to a center in the same blob (mostly)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hdpc.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Density-Peak Clustering Tests ===\n\n");
    const int N=20,D=8;
    float c=1.0f;

    /* two blobs: 10 near (-0.3), 10 near (+0.3) on axis 0 */
    float pts[N*D];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        float sign=(i<5)?-1.0f:((i<15)?1.0f:-1.0f);   /* 3 groups actually:
            [0..4] left, [5..14] right, [15..19] left again — tests 3 centers */
        (void)sign;
        float base=(i<5)?-0.35f:((i<15)?0.35f:-0.30f);
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/5000.0f-0.01f;
            pts[i*D+d]=(d==0)?base+noise:noise;
        }
        /* project into ball */
        float n2=0;for(int d=0;d<D;d++)n2+=pts[i*D+d]*pts[i*D+d];
        if(n2>0.9f){float s=sqrtf(0.9f/n2);for(int d=0;d<D;d++)pts[i*D+d]*=s;}
    }

    int rho[N],assign[N],centers[N];
    CHECK(wubu_hdpc_run(pts,N,D,c,0.05f,rho,assign,centers)==0);

    printf("  g1_centers_found...");
    {
        int nc=0;
        for(int i=0;i<N;i++)if(centers[i])nc++;
        printf("[nc=%d] ",nc);
        CHECK(nc>=2);
    }
    printf("PASS\n");passed++;

    printf("  g2_all_assigned...");
    for(int i=0;i<N;i++)CHECK(assign[i]>=0&&assign[i]<N&&centers[assign[i]]);
    printf("PASS\n");passed++;

    printf("  g3_blob_consistency...");
    {
        /* each point's assigned center should have the same sign of x[0]
         * as the point itself (same blob) */
        int same_blob=0,total=0;
        for(int i=0;i<N;i++){
            total++;
            if((pts[i*D+0]<0)==(pts[assign[i]*D+0]<0))same_blob++;
        }
        float rate=(float)same_blob/total;
        printf("[blob-match=%.2f] ",(double)rate);
        CHECK(rate>=0.8f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
