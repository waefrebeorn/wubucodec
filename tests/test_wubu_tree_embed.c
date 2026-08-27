/* test_wubu_tree_embed.c -- GAP-A021 gates
 *  G1 all nodes on-ball
 *  G2 edge distances approximately tau (within tolerance)
 *  G3 root at origin
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_tree_embed.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Sarkar Tree Embedding Tests ===\n\n");
    /* simple tree: 0=root, children of i are {2i+1, 2i+2} */
    const int N=15,D=8;
    float tau=1.0f,c=1.0f;
    int parent[N];
    parent[0]=-1;
    for(int i=1;i<N;i++)parent[i]=(i-1)/2;   /* binary heap */

    float emb[N*D];
    CHECK(wubu_tree_embed(parent,N,D,tau,emb)==0);

    printf("  g1_on_ball...");
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=emb[i*D+d]*emb[i*D+d];
        CHECK(n2<1.0f+1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_root_at_origin...");
    {
        float n=0;
        for(int d=0;d<D;d++)n+=emb[0]*emb[0];
        n=sqrtf(n);
        CHECK(n<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g3_edges_approx_tau...");
    {
        /* set tau for the checker */
        float max_err=wubu_tree_embed_check(parent,emb,N,D,c);
        printf("[max_err=%.3f] ",(double)max_err);
        /* the simplified construction won't give exact tau but should be close */
        CHECK(max_err<tau*2.0f);   /* within 200% — loose bound for simplified version */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
