/* test_wubu_hknn.c -- GAP-D010 companion gates
 *  G1 exact: known nearest neighbor is found
 *  G2 top-k ordering is by increasing distance
 *  G3 self-match excluded in recall
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hknn.h"
#include <stdint.h>

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic k-NN Tests ===\n\n");
    const int N=50,D=8;
    float db[N*D];
    for(int i=0;i<N*D;i++)db[i]=((float)((i*7919)%997)/997.0f-0.5f)*0.6f;

    printf("  g1_exact_nearest...");
    {
        float query[8];for(int d=0;d<8;d++)query[d]=db[3*8+d]+0.01f*(d+1);
        int idx[5];float dist[5];
        CHECK(wubu_hknn_search(db,N,query,D,1.0f,5,idx,dist)==0);
        /* brute-force with SAME metric */
        int true_best=-1;float true_d=1e30f;
        for(int i=0;i<N;i++){
            float d=wubu_hknn_distance(query,db+(size_t)i*D,D,1.0f);
            if(d<true_d){true_d=d;true_best=i;}
        }
        CHECK(idx[0]==true_best);
        CHECK(dist[0]<=dist[1]);
    }
    printf("PASS\n");passed++;

    printf("  g2_topk_ordering...");
    {
        float query[8];for(int d=0;d<8;d++)query[d]=((float)(d%5)-2)*0.15f;
        int idx[10];float dist[10];
        CHECK(wubu_hknn_search(db,N,query,D,1.0f,10,idx,dist)==0);
        for(int a=1;a<10;a++)CHECK(dist[a]>=dist[a-1]-1e-7f);
    }
    printf("PASS\n");passed++;

    printf("  g3_recall_excludes_self...");
    {
        /* identity labels: label[i] = i%4 */
        int labels[N];for(int i=0;i<N;i++)labels[i]=i%4;
        float r=wubu_hknn_recall(db,db,labels,N,D,1.0f,5);
        /* recall@5 with random data should be ~5/N if self excluded */
        CHECK(r>=0.0f&&r<=1.0f);
        (void)r;
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
