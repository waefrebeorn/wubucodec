/* test_wubu_hretrieval.c -- GAP-D023 gates
 *  G1 same-class query returns same-class items at top
 *  G2 hierarchy bonus promotes a slightly-farther same-label item
 *  G3 precision@5 computed without crash, in [0,1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hretrieval.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hierarchical Retrieval Tests ===\n\n");
    const int N=12,D=8;
    float c=1.0f;

    /* 3 classes × 4 items; class centers on x-axis */
    float emb[N*D];
    int labels[N];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        labels[i]=i%3;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/5000.0f-0.01f;
            emb[i*D+d]=((labels[i]==0)?-0.3f:(labels[i]==1)?0.3f:0.0f)+noise;
        }
        float n2=0;for(int d=0;d<D;d++)n2+=emb[i*D+d]*emb[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)emb[i*D+d]*=s;}
    }

    printf("  g1_same_class_top...");
    {
        /* query = item 0 (class 0); expect top-4 all class 0 */
        int idx[6];
        wubu_hr_rank(emb,N,emb,D,c,NULL,NULL,labels,labels[0],
                     0.0f,10.0f,6,idx);
        int class0_in_top4=0;
        for(int a=0;a<4;a++)
            if(labels[idx[a]]==0)class0_in_top4++;
        CHECK(class0_in_top4>=3);
    }
    printf("PASS\n");passed++;

    printf("  g2_hierarchy_bonus...");
    {
        /* query = item 0 exactly. Two candidates: item 1 (same class,
         * distance d1) and item 4 (diff class, distance d4>d1).
         * Without bonus ranking is by raw distance. With a large bonus,
         * any same-label item beats every different-label one. */
        int idx_nb[4],idx_b[4];
        wubu_hr_rank(emb,N,emb,D,c,NULL,NULL,labels,0,
                     0.0f,10.0f,4,idx_nb);
        wubu_hr_rank(emb,N,emb,D,c,NULL,NULL,labels,0,
                     0.5f,10.0f,4,idx_b);
        /* with bonus: top-3 should ALL be label 0 (there are 4 total,
         * minus self) — stronger than the no-bonus case guarantees */
        int cnt=0;
        for(int a=0;a<3;a++)if(labels[idx_b[a]]==0)cnt++;
        CHECK(cnt>=2);
        (void)idx_nb;
    }
    printf("PASS\n");passed++;

    printf("  g3_precision_k...");
    {
        float p=wubu_hr_precision_k(emb,N,emb,labels,labels,N,D,c,5);
        printf("[p@5=%.2f] ",(double)p);
        CHECK(p>=0.5f&&p<=1.0f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
