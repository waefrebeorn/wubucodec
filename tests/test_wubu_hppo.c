/* test_wubu_hppo.c -- GAP-H015 gates
 *  G1 identical policies → ratio 1, surrogate = mean advantage
 *  G2 clip bounds: huge policy shift doesn't blow up the objective
 *  G3 improvement direction: moving mu toward high-advantage states
 *     yields higher surrogate than moving away
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hppo.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic PPO Tests ===\n\n");
    const int N=4,D=8;
    float c=1.0f,sigma=0.5f,eps=0.2f;

    float states[N*D];
    float adv[4]={1.0f,0.5f,-0.5f,-1.0f};
    for(int i=0;i<N;i++)
        for(int d=0;d<D;d++)states[i*D+d]=((float)((i*8+d)%5)-2)*0.08f;

    float mu_old[D];
    for(int d=0;d<D;d++)mu_old[d]=0.01f*d;

    printf("  g1_identity_ratio...");
    {
        float r=wubu_hp_ratio(mu_old,mu_old,states,D,c,sigma);
        CHECK(fabsf(r-1.0f)<1e-5f);
        float L=wubu_hp_surrogate(mu_old,mu_old,states,adv,N,D,c,sigma,eps);
        /* with ratio=1 everywhere: L = min(A, A) = A, mean = 0 */
        CHECK(fabsf(L)<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_clip_bounds_objective...");
    {
        /* move mu FAR away — ratio explodes but clipped term stays bounded */
        float mu_far[D];
        for(int d=0;d<D;d++)mu_far[d]=mu_old[d]+0.6f;
        float L=wubu_hp_surrogate(mu_far,mu_old,states,adv,N,D,c,sigma,eps);
        /* worst case bound: |L| <= (1+eps)*max|A| */
        CHECK(L<=(1+eps)*1.0f+1e-4f);
        CHECK(L>=-(1+eps)*1.0f-1e-4f);
    }
    printf("PASS\n");passed++;

    printf("  g3_improvement_direction...");
    {
        /* move toward the positive-advantage state vs away from it */
        float toward[D],away[D];
        for(int d=0;d<D;d++){
            toward[d]=mu_old[d]+(states[0*D+d]-mu_old[d])*0.3f;
            away[d]=mu_old[d]-(states[0*D+d]-mu_old[d])*0.3f;
        }
        float Lt=wubu_hp_surrogate(toward,mu_old,states,adv,N,D,c,sigma,eps);
        float La=wubu_hp_surrogate(away,mu_old,states,adv,N,D,c,sigma,eps);
        CHECK(Lt>La);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
