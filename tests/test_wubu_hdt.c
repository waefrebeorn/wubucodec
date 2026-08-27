/* test_wubu_hdt.c -- GAP-D024 gates
 *  G1 tree builds without error, has leaves
 *  G2 training accuracy >= 90% on separable data
 *  G3 predictions deterministic across two calls
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hdt.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Decision Tree Tests ===\n\n");
    /* two separable blobs */
    const int N=20,D=8;
    float c=1.0f;
    float pts[N*D];
    int labels[N];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        labels[i]=i%2;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/5000.0f-0.01f;
            pts[i*D+d]=(labels[i]?0.25f:-0.25f)+noise;
        }
        float n2=0;for(int d=0;d<D;d++)n2+=pts[i*D+d]*pts[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)pts[i*D+d]*=s;}
    }

    WubuHDT tree;
    CHECK(wubu_hdt_build(&tree,pts,labels,N,D,c,4)==0);

    printf("  g1_has_leaves...");
    {
        int leaves=0;
        for(int i=0;i<tree.used;i++)
            if(tree.nodes[i].is_leaf)leaves++;
        CHECK(leaves>=1);
        CHECK(tree.used>0);
    }
    printf("PASS\n");passed++;

    printf("  g2_training_accuracy...");
    {
        int hits=0;
        for(int i=0;i<N;i++){
            int pred=wubu_hdt_predict(&tree,pts+(size_t)i*D);
            if(pred==labels[i])hits++;
        }
        float acc=(float)hits/N;
        printf("[acc=%.2f] ",(double)acc);
        CHECK(acc>=0.9f);
    }
    printf("PASS\n");passed++;

    printf("  g3_deterministic...");
    {
        int p1=wubu_hdt_predict(&tree,pts);
        int p2=wubu_hdt_predict(&tree,pts);
        CHECK(p1==p2);
    }
    printf("PASS\n");passed++;

    wubu_hdt_free(&tree);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
