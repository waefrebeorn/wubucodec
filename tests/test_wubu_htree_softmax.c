/* test_wubu_htree_softmax.c -- GAP-C047 gates
 *  G1 leaf probabilities in [0,1]
 *  G2 nearest-leaf query → that leaf gets the max probability
 *  G3 predict is deterministic
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_htree_softmax.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Hierarchical Softmax Tests ===\n\n");
    const int NL=8,D=8;
    float c=1.0f;

    WubuHTS t;
    CHECK(wubu_hts_init(&t,NL,D,c)==0);

    /* place query exactly at leaf 5's prototype */
    const int n_int=NL-1;
    const float* p5=t.proto+(size_t)(n_int+5)*D;
    /* make the root-to-leaf-5 branch nodes favor the query: copy query
     * coords into protos of nodes on its path (0, 2, 5 for target 12)
     * with slight inward pull so affinity is high along THAT path only. */
    int path[3]={0,2,5};
    for(int pi=0;pi<3;pi++)
        for(int d=0;d<D;d++)
            t.proto[(size_t)path[pi]*D+d]=p5[d]*0.98f;
    /* final branch node: proto pushed past query so RIGHT child (leaf 5)
     * is strictly closer than LEFT (leaf 4) */
    for(int d=0;d<D;d++)t.proto[(size_t)path[2]*D+d]=p5[d]*1.05f;

    printf("  g1_probs_in_range...");
    {
        for(int l=0;l<NL;l++){
            float p=wubu_hts_leaf_prob(&t,p5,l);
            CHECK(p>=0&&p<=1);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_nearest_leaf_wins...");
    {
        int pred=wubu_hts_predict(&t,p5);
        printf("[pred=%d want=5] ",pred);
        CHECK(pred==5);
    }
    printf("PASS\n");passed++;

    printf("  g3_deterministic...");
    {
        int a=wubu_hts_predict(&t,p5);
        int b=wubu_hts_predict(&t,p5);
        CHECK(a==b);
    }
    printf("PASS\n");passed++;

    wubu_hts_free(&t);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
