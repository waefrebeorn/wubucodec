/* test_wubu_hsupcon.c -- GAP-D031 gates
 *  G1 perfect clustering (same-class together, far apart) → low loss
 *  G2 scrambled labels → high loss
 *  G3 loss(good) < loss(bad)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hsupcon.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic SupCon Tests ===\n\n");
    const int N=8,D=4;
    float c=1.0f,tau=0.5f;

    /* two clusters of 4 on opposite sides */
    float z[N*D];
    for(int i=0;i<N;i++)
        for(int d=0;d<D;d++)
            z[i*D+d]=((i<4)?-0.5f:0.5f)*(((d+1)%2)?0.9f:0.05f);

    printf("  g1_good_arrangement_low_loss...");
    {
        int labels[8]={0,0,0,0,1,1,1,1};
        float L=wubu_hsc_loss(z,labels,NULL,2,N,D,c,tau);
        printf("[L=%.4f] ",(double)L);
        CHECK(L<1.5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_bad_labels_high_loss...");
    {
        /* alternating labels = positives scattered across clusters */
        int labels[8]={0,1,0,1,0,1,0,1};
        float L=wubu_hsc_loss(z,labels,NULL,2,N,D,c,tau);
        printf("[L=%.4f] ",(double)L);
        CHECK(L>1.0f);
    }
    printf("PASS\n");passed++;

    printf("  g3_good_loses_less...");
    {
        int good[8]={0,0,0,0,1,1,1,1};
        int bad[8]={0,1,0,1,0,1,0,1};
        float Lg=wubu_hsc_loss(z,good,NULL,2,N,D,c,tau);
        float Lb=wubu_hsc_loss(z,bad,NULL,2,N,D,c,tau);
        CHECK(Lg<Lb);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
