/* test_wubu_hbandit.c -- GAP-H012 gates
 *  G1 posteriors start uniform (mean 0.5)
 *  G2 reward updates move the mean toward the truth
 *  G3 bandit learns: after training, best action's mean exceeds others
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hbandit.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Contextual Bandit Tests ===\n\n");
    const int NA=3,D=8;
    float c=1.0f;

    WubuHB b;
    CHECK(wubu_hb_init(&b,NA,D,1.0f,42u)==0);

    /* fixed context near action 0's prototype region */
    float ctx[D];
    for(int d=0;d<D;d++)ctx[d]=b.proto[0*D+d]+((float)(d%3)-1)*0.02f;
    float n2=0;for(int d=0;d<D;d++)n2+=ctx[d]*ctx[d];
    if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)ctx[d]*=s;}

    printf("  g1_uniform_start...");
    for(int a=0;a<NA;a++)
        CHECK(fabsf(wubu_hb_mean_reward(&b,a)-0.5f)<1e-5f);
    printf("PASS\n");passed++;

    printf("  g2_reward_updates_mean...");
    {
        float m0=wubu_hb_mean_reward(&b,0);
        for(int i=0;i<20;i++)wubu_hb_update(&b,0,ctx,1.0f,c);
        float m1=wubu_hb_mean_reward(&b,0);
        CHECK(m1>m0+0.2f);
    }
    printf("PASS\n");passed++;

    printf("  g3_best_action_learned...");
    {
        /* train: action 0 always rewards at this context, others don't */
        for(int i=0;i<50;i++){
            wubu_hb_update(&b,0,ctx,1.0f,c);
            wubu_hb_update(&b,1,ctx,0.0f,c);
            wubu_hb_update(&b,2,ctx,0.0f,c);
        }
        float m0=wubu_hb_mean_reward(&b,0);
        float m1=wubu_hb_mean_reward(&b,1);
        float m2=wubu_hb_mean_reward(&b,2);
        printf("[m0=%.2f m1=%.2f m2=%.2f] ",(double)m0,(double)m1,(double)m2);
        CHECK(m0>m1&&m0>m2);
    }
    printf("PASS\n");passed++;

    wubu_hb_free(&b);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
