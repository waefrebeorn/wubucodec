/* test_wubu_av_fidelity.c -- GAP-E005 gates
 *  G1 loud segment gets higher weight than silent one
 *  G2 weights clamp to [w_min, 1]
 *  G3 loss increases with distortion and with bits
 */
#include <stdio.h>
#include <math.h>
#include "wubu_av_fidelity.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
int main(void){
    printf("=== WuBuMath AV Fidelity Tests ===\n\n");
    const int T=4;
    float be[T*5];   /* band energies: t=1 loud, t=2 silent */
    for(int t=0;t<T;t++)
        for(int b=0;b<5;b++)
            be[t*5+b]=(t==1)?1.0f:(t==2?0.0f:0.4f);
    float w[T];
    wubu_av_weights(be,w,T,0.5f,0.8f,0.2f);

    printf("  g1_loud_gets_weight...");
    CHECK(w[1]>w[2]);
    printf("PASS\n");passed++;

    printf("  g2_clamped...");
    for(int t=0;t<T;t++) CHECK(w[t]>=0.2f&&w[t]<=1.0f);
    CHECK(fabsf(w[2]-0.5f)<1e-6f);   /* silent sits at alpha base */
    CHECK(fabsf(w[1]-1.0f)<1e-6f);   /* loud saturates */
    printf("PASS\n");passed++;

    printf("  g3_loss_monotone...");
    float dist[T]; for(int i=0;i<T;i++)dist[i]=0.1f;
    float L1=wubu_av_fidelity_loss(dist,w,T,100.0f,0.001f);
    dist[1]=0.4f;   /* worse distortion at the loud spot hurts more */
    float L2=wubu_av_fidelity_loss(dist,w,T,100.0f,0.001f);
    float L3=wubu_av_fidelity_loss(dist,w,T,300.0f,0.001f);
    CHECK(L2>L1);
    CHECK(L3>L2);
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
