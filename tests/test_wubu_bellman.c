/* test_wubu_bellman.c -- GAP-H016 gates (the 100th closure)
 *  G1 contraction: sup-diff shrinks by gamma for random V pairs
 *  G2 monotonicity: pointwise order preserved by T
 *  G3 fixed point: value iteration converges, T(V*)=V*
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_bellman.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Bellman Operator Tests (100th gap) ===\n\n");
    const int S=6,A=2;
    float gamma=0.9f;

    /* ring MDP: state s -> s+1 via action 1, stay via action 0 */
    int8_t next[S*A];
    float R[S*A];
    for(int s=0;s<S;s++){
        next[s*A+0]=(int8_t)s;              /* stay */
        next[s*A+1]=(int8_t)((s+1)%S);      /* advance */
    }
    for(int s=0;s<S;s++){
        R[s*A+0]=0.0f;
        R[s*A+1]=((s+1)%S==0)?1.0f:0.01f;   /* reward at wraparound */
    }

    printf("  g1_contraction...");
    {
        int ok=1;
        for(unsigned seed=1u;seed<=20u;seed++)
            if(!wubu_bellman_contraction_check(next,R,S,A,gamma,seed)){ok=0;break;}
        CHECK(ok);
    }
    printf("PASS\n");passed++;

    printf("  g2_monotone...");
    {
        float V1[6],V2[6],T1[6],T2[6];
        unsigned rs=99u;
        for(int i=0;i<S;i++){
            rs=rs*1103515245u+12345u;
            V1[i]=((float)((rs>>16)%1000))/2000.0f;
            V2[i]=V1[i]+(float)((rs>>8)%500)/1000.0f;   /* >= V1 */
        }
        wubu_bellman_apply(V1,next,R,S,A,gamma,T1);
        wubu_bellman_apply(V2,next,R,S,A,gamma,T2);
        for(int s=0;s<S;s++)CHECK(T2[s]>=T1[s]-1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g3_fixed_point...");
    {
        float V[6],TV[6];
        for(int i=0;i<S;i++)V[i]=0;
        /* value iteration to convergence */
        for(int it=0;it<500;it++){
            wubu_bellman_apply(V,next,R,S,A,gamma,TV);
            memcpy(V,TV,sizeof(float)*S);
        }
        /* T(V*) == V* */
        wubu_bellman_apply(V,next,R,S,A,gamma,TV);
        CHECK(wubu_bellman_supdiff(V,TV,S)<1e-4f);
        /* sanity: value of the state right before reward ≈ 1 */
        CHECK(V[5]>0.9f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
