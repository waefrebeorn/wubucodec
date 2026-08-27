/* test_wubu_hier.c -- GAP-D015 gates
 *  G1 classify returns valid leaf index
 *  G2 accuracy on data near leaf prototypes beats chance
 *  G3 confidence (margin) is positive for clear cases
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hier.h"
#include "wubu_tree_embed.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hierarchical Classification Tests ===\n\n");
    /* binary tree: 7 nodes, leaves = {3,4,5,6} */
    const int N=7,N_LEAF=4,D=8;
    float c=1.0f;
    int parent[7]={-1,0,0,1,1,2,2};
    int leaf_idx[4]={3,4,5,6};

    WubuHier h;
    CHECK(wubu_hier_init(&h,parent,N,leaf_idx,N_LEAF,D,c)==0);

    printf("  g1_valid_classification...");
    {
        float x[8];
        for(int d=0;d<D;d++)x[d]=h.proto[3*8+d]+0.05f;  /* near leaf 3 */
        int pred=wubu_hier_classify(&h,x,NULL);
        CHECK(pred>=0&&pred<N_LEAF);
        CHECK(pred==0);  /* should map to leaf index 0 (= node 3) */
    }
    printf("PASS\n");passed++;

    printf("  g2_accuracy_beats_chance...");
    {
        /* generate noisy samples around each leaf prototype */
        float xs[N_LEAF*10*8];
        int truth[N_LEAF*10];
        unsigned rs=42u;
        for(int li=0;li<N_LEAF;li++){
            int node=leaf_idx[li];
            for(int s=0;s<10;s++){
                int idx=li*10+s;
                truth[idx]=li;
                for(int d=0;d<D;d++){
                    rs=rs*1103515245u+12345u;
                    float n=(float)((rs>>16)%100)/1000.0f-0.05f;
                    xs[idx*8+d]=h.proto[node*8+d]+n;
                }
            }
        }
        float acc=wubu_hier_accuracy(&h,xs,truth,N_LEAF*10);
        printf("[acc=%.2f] ",(double)acc);
        CHECK(acc>=0.75f);   /* chance = 25% */
    }
    printf("PASS\n");passed++;

    printf("  g3_confidence_positive...");
    {
        int conf;
        float x[8];
        for(int d=0;d<8;d++)x[d]=h.proto[4*8+d];  /* exactly at leaf 4 */
        wubu_hier_classify(&h,x,&conf);
        CHECK(conf>0);   /* margin over second-best should be positive */
    }
    printf("PASS\n");passed++;

    wubu_hier_free(&h);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
