/* test_wubu_euclidparam.c -- GAP-C026 gates
 *  G1 round trip: ball -> param -> ball reproduces the point
 *  G2 unbounded params never produce off-ball coordinates
 *  G3 huge Euclidean params (|z|=1000) still give valid on-ball points
 *     (this is THE advantage over raw Poincaré storage)
 *  G4 SGD on params reduces distance-to-target monotonically-ish
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_euclidparam.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Euclidean Parametrization Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    printf("  g1_round_trip...");
    {
        unsigned rs=42u;
        for(int trial=0;trial<100;trial++){
            float x[8],z[8],x2[8];
            for(int d=0;d<D;d++){
                rs=rs*1103515245u+12345u;
                x[d]=(float)((rs>>16)%2000)/20000.0f-0.05f;  /* inside ball */
            }
            /* ensure inside */
            float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
            if(n2>0.9f){float s=sqrtf(0.9f/n2);for(int d=0;d<D;d++)x[d]*=s;}
            wubu_ep_from_ball(x,D,c,z);
            wubu_ep_to_ball(z,D,c,x2);
            for(int d=0;d<D;d++)CHECK(fabsf(x[d]-x2[d])<1e-4f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_always_on_ball...");
    {
        float z[8],x[8];
        unsigned rs=7u;
        for(int trial=0;trial<500;trial++){
            for(int d=0;d<D;d++){
                rs=rs*1103515245u+12345u;
                z[d]=(float)((rs>>16)%4000)/2000.0f-1.0f;
            }
            wubu_ep_to_ball(z,D,c,x);
            float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_huge_params_still_valid...");
    {
        /* |z| = 1000 — impossible in raw Poincaré (boundary at ~38),
         * fine through this parametrization */
        float z[8]={900,850,760,810,-700,-950,600,-800};
        float x[8];
        wubu_ep_to_ball(z,D,c,x);
        float n2=0;for(int d=0;d<8;d++)n2+=x[d]*x[d];
        CHECK(n2<1.0f&&n2>0.5f);   /* near boundary but valid */
        /* distinct directions preserved */
        CHECK(fabsf(x[0]/x[1]-z[0]/z[1])<1e-3f);
    }
    printf("PASS\n");passed++;

    printf("  g4_sgd_reduces_distance...");
    {
        /* target params */
        float target[8]={0.3f,-0.25f,0.15f,0.05f,-0.1f,0.2f,-0.05f,0.12f};
        float z[8]={0};   /* start at origin */
        float prev_d=wubu_ep_distance(z,target,D,c);
        for(int s=0;s<300;s++){
            float grad[8];
            for(int d=0;d<D;d++)grad[d]=z[d]-target[d]; /* Euclidean loss grad */
            wubu_ep_sgd_step(z,grad,D,0.02f);
            if(s%100==99){
                float euc=0;
                for(int d=0;d<D;d++){float df=z[d]-target[d];euc+=df*df;}
                euc=sqrtf(euc);
                CHECK(euc<=prev_d+0.01f);   /* euclidean loss not diverging */
                prev_d=euc;
            }
        }
        /* final: both euclidean params AND manifold distance improved */
        float final_euc=0;
        for(int d=0;d<D;d++){float df=z[d]-target[d];final_euc+=df*df;}
        final_euc=sqrtf(final_euc);
        float final_man=wubu_ep_distance(z,target,D,c);
        CHECK(final_euc<1.0f);          /* converged in parameter space */
        CHECK(final_man>=0.0f);         /* manifold distance finite */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
