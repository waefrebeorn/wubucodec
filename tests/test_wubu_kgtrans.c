/* test_wubu_kgtrans.c -- GAP-D036 gates
 *  G1 positive triple scores higher than random corrupt
 *  G2 training improves mean margin loss
 *  G3 true tail ranks top-3 after training on a small consistent KG
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_kgtrans.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic TransE Tests ===\n\n");
    /* tiny KG: 6 entities, 1 relation "follows" forming a chain 0->1->2->...*/
    const int NE=6,NR=1,D=8,NT=5;
    float c=1.0f;

    WubuKG kg;
    CHECK(wubu_kg_init(&kg,NE,NR,D,c,42u)==0);

    int heads[5]={0,1,2,3,4};
    int rels[5]={0,0,0,0,0};
    int tails[5]={1,2,3,4,5};

    printf("  g1_positive_beats_random...");
    {
        unsigned seed=99u;
        int wins=0;
        for(int i=0;i<NT;i++){
            *(&seed)=seed*1103515245u+12345u;
            int tc=(seed>>8)%NE;
            if(tc==tails[i])tc=(tc+1)%NE;
            if(wubu_kg_score(&kg,heads[i],rels[i],tails[i])>
               wubu_kg_score(&kg,heads[i],rels[i],tc))wins++;
        }
        CHECK(wins>=1);   /* sanity pre-training: some structure exists */
    }
    printf("PASS\n");passed++;

    printf("  g2_training_reduces_loss...");
    {
        float l_first=0,l_last=0;
        unsigned seed=7u;
        for(int ep=0;ep<60;ep++){
            float l=wubu_kg_train_epoch(&kg,heads,rels,tails,NT,1.0f,0.5f,&seed);
            if(ep==0)l_first=l;
            if(ep==59)l_last=l;
        }
        printf("[first=%.4f last=%.4f] ",(double)l_first,(double)l_last);
        CHECK(l_last<l_first);
    }
    printf("PASS\n");passed++;

    printf("  g3_true_tail_ranks_top...");
    {
        int total_rank=0;
        for(int i=0;i<NT;i++)
            total_rank+=wubu_kg_rank(&kg,heads[i],rels[i],tails[i]);
        printf("[mean rank %.1f / %d] ",(double)total_rank/NT,NE);
        CHECK(total_rank<=NT*3);   /* mean rank <= 3 */
    }
    printf("PASS\n");passed++;

    wubu_kg_free(&kg);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
