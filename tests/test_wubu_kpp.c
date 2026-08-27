/* test_wubu_kpp.c -- GAP-D033 gates
 *  G1 distinct centers (no duplicates)
 *  G2 two-blob data: seeds land in BOTH blobs
 *  G3 deterministic given the same seed
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_kpp.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== k-means++ Seeding Tests ===\n\n");
    const int N=10,D=4,K=3;
    float c=1.0f;

    /* two blobs of 5 on opposite sides + jitter */
    float pts[N*D];
    unsigned rs=11u;
    for(int i=0;i<N;i++)
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/5000.0f-0.01f;
            pts[i*D+d]=((i<5)?-0.6f:0.6f)*(((d+1)%2)?1.0f:0.05f)+noise;
        }

    printf("  g1_distinct_centers...");
    {
        unsigned seed=42u;
        int centers[3];
        CHECK(wubu_kpp_seed(pts,N,D,K,c,&seed,centers)==0);
        for(int a=0;a<K;a++)
            for(int b=a+1;b<K;b++)
                CHECK(centers[a]!=centers[b]);
    }
    printf("PASS\n");passed++;

    printf("  g2_both_blobs_seeded...");
    {
        int hits=0;
        /* run several seeds; expect most runs to cover both blobs */
        for(unsigned s=1;s<=20;s++){
            unsigned seed=s;
            int centers[3];
            wubu_kpp_seed(pts,N,D,2,c,&seed,centers);
            int in_low=0,in_hi=0;
            for(int m=0;m<2;m++){
                if(pts[centers[m]*D]<0)in_low++;
                else in_hi++;
            }
            if(in_low&&in_hi)hits++;
        }
        printf("[both-blob coverage %d/20] ",hits);
        CHECK(hits>=15);
    }
    printf("PASS\n");passed++;

    printf("  g3_deterministic...");
    {
        unsigned s1=7u,s2=7u;
        int c1[3],c2[3];
        wubu_kpp_seed(pts,N,D,K,c,&s1,c1);
        wubu_kpp_seed(pts,N,D,K,c,&s2,c2);
        for(int m=0;m<K;m++)CHECK(c1[m]==c2[m]);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
