/* test_wubu_hkmedoids.c -- GAP-D025 gates
 *  G1 medoids are actual data points (indices in [0,n))
 *  G2 robustness: planted outlier does NOT become a medoid
 *  G3 clusters recovered with high accuracy on two blobs
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hkmedoids.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic K-Medoids Tests ===\n\n");
    /* 12 tight points + 1 far outlier; k=2 should pick medoids from tight pts */
    const int N=13,D=8,K=2;
    float c=1.0f;
    float pts[N*D];
    unsigned rs=42u;
    for(int i=0;i<12;i++){
        float base=(i%2)?0.25f:-0.25f;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/5000.0f-0.01f;
            pts[i*D+d]=(d==0)?base+noise:noise;
        }
    }
    /* outlier near boundary */
    for(int d=0;d<D;d++)pts[12*D+d]=(d==0)?-0.85f:((float)(d%3)-1)*0.02f;

    int medoids[K],assign[N];
    CHECK(wubu_hkmedoids(pts,N,D,K,c,50,medoids,assign)==0);

    printf("  g1_medoids_are_data_points...");
    for(int m=0;m<K;m++)CHECK(medoids[m]>=0&&medoids[m]<N);
    printf("PASS\n");passed++;

    printf("  g2_outlier_not_medoid...");
    {
        int outlier_is_med=0;
        for(int m=0;m<K;m++)if(medoids[m]==12)outlier_is_med=1;
        CHECK(!outlier_is_med);
    }
    printf("PASS\n");passed++;

    printf("  g3_clusters_recovered...");
    {
        /* ground truth: even i -> class A(base +0.25), odd i -> class B(-0.25)
         * for i<12; outlier 12 excluded from accuracy */
        int correct=0,total=0;
        /* map medoid identity to cluster by its own base sign */
        int med_class[K];
        for(int m=0;m<K;m++)
            med_class[m]=(pts[medoids[m]*D+0]>0)?1:0;
        for(int i=0;i<12;i++){
            int truth=(pts[i*D+0]>0)?1:0;
            if(med_class[assign[i]]==truth)correct++;
            total++;
        }
        float acc=(float)correct/total;
        printf("[acc=%.2f] ",(double)acc);
        CHECK(acc>=0.9f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
