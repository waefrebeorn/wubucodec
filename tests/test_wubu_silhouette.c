/* test_wubu_silhouette.c -- GAP-D034 gates
 *  G1 correct clustering → high positive silhouette
 *  G2 wrong clustering → lower (possibly negative)
 *  G3 good > bad
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_silhouette.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Silhouette Tests ===\n\n");
    const int N=8,D=4,K=2;
    float c=1.0f;

    /* two tight blobs on opposite sides */
    float pts[N*D];
    unsigned rs=13u;
    for(int i=0;i<N;i++)
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/20000.0f-0.0025f;
            pts[i*D+d]=((i<4)?-0.55f:0.55f)*(((d+1)%2)?1.0f:0.05f)+noise;
        }

    printf("  g1_correct_high...");
    {
        int assign[8]={0,0,0,0,1,1,1,1};
        float s=wubu_sil_score(pts,assign,N,D,K,c);
        printf("[s=%.3f] ",(double)s);
        CHECK(s>0.5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_wrong_low...");
    {
        /* half of each blob assigned to each cluster */
        int assign[8]={0,1,0,1,0,1,0,1};
        float s=wubu_sil_score(pts,assign,N,D,K,c);
        printf("[s=%.3f] ",(double)s);
        CHECK(s<0.5f);
    }
    printf("PASS\n");passed++;

    printf("  g3_good_beats_bad...");
    {
        int good[8]={0,0,0,0,1,1,1,1};
        int bad[8]={0,1,0,1,0,1,0,1};
        float sg=wubu_sil_score(pts,good,N,D,K,c);
        float sb=wubu_sil_score(pts,bad,N,D,K,c);
        CHECK(sg>sb);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
