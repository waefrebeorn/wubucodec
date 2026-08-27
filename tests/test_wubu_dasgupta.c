/* test_wubu_dasgupta.c -- GAP-D027 gates
 *  G1 good tree (high-similarity pairs merge late = low LCA) scores
 *     LOWER than bad tree (high-similarity pairs split early)
 *  G2 cost is non-negative
 */
#include <stdio.h>
#include <stdlib.h>
#include "wubu_dasgupta.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Dasgupta Cost Tests ===\n\n");
    /* 4 leaves: {0,1} very similar, {2,3} similar, cross low */
    const int NN=4;   /* leaves */
    float w[16]={0};
    w[0*4+1]=w[1*4+0]=9.0f;
    w[2*4+3]=w[3*4+2]=8.0f;
    /* cross similarities small */
    for(int i=0;i<2;i++)
        for(int j=2;j<4;j++){
            w[i*4+j]=w[j*4+i]=1.0f;
        }

    /* GOOD tree: 0,1 merge first; 2,3 merge first; roots join last.
     * Nodes: 0..3 leaves, 4={0,1}, 5={2,3}, 6=root={4,5} */
    int gl[7]={-1,-1,-1,-1,0,2,4};
    int gr[7]={-1,-1,-1,-1,1,3,5};
    int gsz[7]={1,1,1,1,2,2,4};

    /* BAD tree: splits similar pairs early. 4={0,2}, 5={1,3}, 6=root */
    int bl[7]={-1,-1,-1,-1,0,1,4};
    int br[7]={-1,-1,-1,-1,2,3,5};
    int bsz[7]={1,1,1,1,2,2,4};

    printf("  g2_nonneg...");
    {
        double cg=wubu_dasgupta_cost(gl,gr,gsz,7,w,NN);
        CHECK(cg>=0);
    }
    printf("PASS\n");passed++;

    printf("  g1_good_tree_cheaper...");
    {
        double c_good=wubu_dasgupta_cost(gl,gr,gsz,7,w,NN);
        double c_bad=wubu_dasgupta_cost(bl,br,bsz,7,w,NN);
        printf("[good=%.1f bad=%.1f] ",c_good,c_bad);
        CHECK(c_good<c_bad);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
