/* test_wubu_hkmeans.c -- GAP-C020 gates: hyperbolic k-means
 *  G1 well-separated clusters recovered (accuracy >= 80%)
 *  G2 all centroids inside the ball
 *  G3 every point assigned exactly once
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hkmeans.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic K-Means Tests ===\n\n");
    const int N=60,K=3,D=8;
    float c=1.0f;

    /* generate 3 well-separated clusters */
    float pts[N*D];int truth[N];
    unsigned rs=42u;
    float centers[3][8]={
        {-0.4f,-0.1f, 0.1f,-0.2f, 0.05f, 0.15f,-0.05f, 0.1f},
        { 0.4f, 0.1f,-0.1f, 0.2f,-0.05f,-0.15f, 0.05f,-0.1f},
        { 0.0f, 0.3f, 0.25f,0.0f, 0.2f, -0.1f,  0.15f,-0.05f}};
    for(int i=0;i<N;i++){
        int cl=i%K;truth[i]=cl;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float n=(float)((rs>>16)%2000)/2000.0f-0.5f;
            pts[(size_t)i*D+d]=centers[cl][d]+n*0.08f;
        }
        /* project into ball: scale to |x|<0.9 */
        float n2=0;for(int d=0;d<D;d++)n2+=pts[i*D+d]*pts[i*D+d];
        if(n2>0.81f){float s2=0.9f/sqrtf(n2);
            for(int d=0;d<D;d++)pts[i*D+d]*=s2;}
    }

    int assign[N];float cent[K*D];
    int iters=wubu_hkmeans(pts,N,D,K,c,100,1e-5f,assign,cent);
    printf("  iterations=%d\n",iters);

    printf("  g2_centroids_on_ball...");
    for(int k=0;k<K;k++){
        float n2=0;for(int d=0;d<D;d++)n2+=cent[k*D+d]*cent[k*D+d];
        CHECK(n2<1.0f);
    }
    printf("PASS\n");passed++;

    printf("  g3_all_assigned...");
    for(int i=0;i<N;i++)CHECK(assign[i]>=0&&assign[i]<K);
    printf("PASS\n");passed++;

    printf("  g1_clusters_recovered...");
    /* best permutation of cluster labels vs ground truth */
    int best_acc=0;
    int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for(int p=0;p<6;p++){
        int acc=0;
        for(int i=0;i<N;i++){
            int mapped=perm[p][assign[i]];
            if(mapped==truth[i])acc++;
        }
        if(acc>best_acc)best_acc=acc;
    }
    float acc_f=(float)best_acc/N;
    printf("[acc=%.2f] ",(double)acc_f);
    CHECK(acc_f>=0.8f);
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
