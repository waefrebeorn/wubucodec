/* test_wubu_hiermerge.c -- GAP-D016 gates
 *  G1 tree has exactly n-1 internal nodes, all alive except merged
 *  G2 two well-separated pairs merge WITHIN pairs before ACROSS
 *  G3 root prototype is on-ball
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hiermerge.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Hierarchical Clustering Tests ===\n\n");
    const int N=4,D=8;
    float c=1.0f;

    /* two tight pairs, far apart:
     * pair A = {p0,p1} near (-0.3, ...), pair B = {p2,p3} near (+0.3, ...) */
    float pts[N*D];
    for(int d=0;d<D;d++){
        pts[0*D+d]=-0.3f+((float)(d%3)-1)*0.01f;
        pts[1*D+d]=-0.28f+((float)(d%4)-1)*0.01f;
        pts[2*D+d]= 0.30f+((float)(d%3)-1)*0.01f;
        pts[3*D+d]= 0.32f+((float)(d%4)-1)*0.01f;
    }
    /* project into ball */
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=pts[i*D+d]*pts[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)pts[i*D+d]*=s;}
    }

    WubuHMNode tree[2*N-1];
    int root=wubu_hm_cluster(pts,N,D,c,tree);
    printf("  root=%d\n",root);

    printf("  g1_structure...");
    {
        int alive_count=0;
        for(int i=0;i<2*N-1;i++)if(tree[i].alive)alive_count++;
        CHECK(alive_count==1);          /* only root remains */
        CHECK(tree[root].size==N);
        CHECK(root>=N&&root<2*N-1);     /* root is internal */
    }
    printf("PASS\n");passed++;

    printf("  g2_pairs_merge_first...");
    {
        /* the first internal node (index N) should join two points from
         * the SAME pair (both near -0.3 or both near +0.3) */
        WubuHMNode* first=&tree[N];
        CHECK(first->left>=0&&first->right>=0);
        float l=pts[first->left*D];   /* first coord distinguishes pairs */
        float r=pts[first->right*D];
        CHECK((l<0&&r<0)||(l>0&&r>0));   /* same sign = same pair */
    }
    printf("PASS\n");passed++;

    printf("  g3_root_on_ball...");
    {
        float n2=0;
        for(int d=0;d<D;d++)n2+=tree[root].proto[d]*tree[root].proto[d];
        CHECK(n2<1.0f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
