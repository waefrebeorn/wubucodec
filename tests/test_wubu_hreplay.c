/* test_wubu_hreplay.c -- GAP-H013 gates
 *  G1 buffer fills to capacity, wraps correctly
 *  G2 high-TD samples drawn more often than low-TD (priority works)
 *  G3 diversity bonus: isolated sample outranks clustered one at equal TD
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hreplay.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Experience Replay Tests ===\n\n");
    const int CAP=8,D=4;
    float c=1.0f;

    WubuRP r;
    CHECK(wubu_rp_init(&r,CAP,D,c,0.5f)==0);

    printf("  g1_fill_and_wrap...");
    {
        for(int i=0;i<CAP+3;i++){
            float s[D];
            for(int d=0;d<D;d++)s[d]=(float)(i%5)*0.01f+d*0.001f;
            wubu_rp_add(&r,s,1.0f);
        }
        CHECK(r.len==CAP);
        CHECK(r.head==3);   /* wrapped */
    }
    printf("PASS\n");passed++;

    printf("  g2_high_td_sampled_more...");
    {
        wubu_rp_free(&r);
        wubu_rp_init(&r,CAP,D,c,0.0f);   /* no diversity bonus: pure TD */
        /* two distinct states, very different TD errors */
        float sA[4]={0.1f,0.0f,0.0f,0.0f};
        float sB[4]={-0.6f,0.0f,0.0f,0.0f};
        wubu_rp_add(&r,sA,0.01f);
        wubu_rp_add(&r,sB,2.0f);
        unsigned seed=7u;
        int countA=0,countB=0;
        for(int i=0;i<200;i++){
            int idx=wubu_rp_sample(&r,&seed);
            if(idx==0)countA++;
            else if(idx==1)countB++;
        }
        CHECK(countB>countA*3);
    }
    printf("PASS\n");passed++;

    printf("  g3_diversity_bonus...");
    {
        wubu_rp_free(&r);
        wubu_rp_init(&r,CAP,D,c,10.0f);
        /* three tightly-clustered states with tiny TD + one isolated with SAME tiny TD */
        float clus[3][4]={{0.10f,0,0,0},{0.11f,0.005f,0,0},{0.105f,-0.004f,0.002f,0}};
        float iso[4]={-0.7f,0.05f,0.02f,0.0f};
        for(int i=0;i<3;i++)wubu_rp_add(&r,clus[i],0.01f);
        wubu_rp_add(&r,iso,0.01f);
        /* the isolated one should have the highest priority */
        float best=r.prio[0];int bi=0;
        for(int i=1;i<r.len;i++)if(r.prio[i]>best){best=r.prio[i];bi=i;}
        CHECK(bi==3);
    }
    printf("PASS\n");passed++;

    wubu_rp_free(&r);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
