/* test_wubu_graph2ball.c -- GAP-D022 gates
 *  G1 walks stay on graph nodes
 *  G2 training reduces average loss over epochs
 *  G3 adjacent nodes end closer than non-adjacent (separation > 0)
 */
#include <stdio.h>
#include <stdlib.h>
#include "wubu_graph2ball.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Graph-to-Ball Embedding Tests ===\n\n");
    /* two 4-cliques bridged by node 3-4 edge */
    const int N=8,D=8;
    /* edges: cliques {0,1,2,3} and {4,5,6,7}, plus bridge 3-4 */
    int edges[][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3},
                    {4,5},{4,6},{4,7},{5,6},{5,7},{6,7},
                    {3,4}};
    const int NE=13;
    int deg[8]={0};
    for(int i=0;i<NE;i++){deg[edges[i][0]]++;deg[edges[i][1]]++;}
    int adj_ptr[9]={0};
    for(int i=0;i<8;i++)adj_ptr[i+1]=adj_ptr[i]+deg[i];
    int adj_idx[26]={0};
    int fill[8]={0};
    for(int i=0;i<NE;i++){
        int a=edges[i][0],b=edges[i][1];
        adj_idx[adj_ptr[a]+fill[a]++]=b;
        adj_idx[adj_ptr[b]+fill[b]++]=a;
    }
    WubuG2B g;
    CHECK(wubu_g2b_init(&g,N,D,1.0f,0.02f,42u)==0);

    printf("  g1_walks_valid...");
    {
        for(int start=0;start<N;start+=3){
            int walk[12];
            wubu_g2b_walk(adj_idx,adj_ptr,N,start,12,walk);
            for(int s=0;s<12;s++){
                CHECK(walk[s]>=0&&walk[s]<N);
                if(s>0){
                    /* consecutive walk entries must be adjacent */
                    int a=walk[s-1],b=walk[s],adjacent=0;
                    for(int j=adj_ptr[a];j<adj_ptr[a+1];j++)
                        if(adj_idx[j]==b){adjacent=1;break;}
                    CHECK(adjacent);
                }
            }
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_loss_decreases...");
    {
        float l_first=-1,l_last=-1;
        for(int epoch=0;epoch<200;epoch++){
            float l=wubu_g2b_train(&g,adj_idx,adj_ptr,10,2,2,3);
            if(epoch==0)l_first=l;
            if(epoch==29)l_last=l;
        }
        printf("[%.4f->%.4f] ",(double)l_first,(double)l_last);
        CHECK(l_last<l_first);
    }
    printf("PASS\n");passed++;

    printf("  g3_adjacency_separation...");
    {
        float sep=wubu_g2b_separation(&g,adj_idx,adj_ptr);
        printf("[sep=%.4f] ",(double)sep);
        /* positive = edges closer than non-edges */
        CHECK(sep>0.0f);
    }
    printf("PASS\n");passed++;

    wubu_g2b_free(&g);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
