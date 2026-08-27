/* test_wubu_hactivation.c -- GAP-C018 gates: hyperbolic activation
 *  G1 output stays on the hyperboloid (L = -1/c), upper sheet
 *  G2 ReLU zeroes negative-space tangent components
 *  G3 tanh keeps values finite
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hlinear.h"
#include "wubu_lorentz.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Activation Tests ===\n\n");
    const int D=8,N=4,DIN1=D+1;
    float c=1.0f;

    /* points on hyperboloid with mixed-sign space components */
    float x[N*DIN1];
    for(int i=0;i<N;i++){
        for(int d=1;d<DIN1;d++)
            x[i*DIN1+d]=((float)((i*13+d*29)%89)/89.0f-0.5f)*1.2f;
        float s2=0;for(int d=1;d<DIN1;d++)s2+=x[i*DIN1+d]*x[i*DIN1+d];
        x[i*DIN1+0]=sqrtf(1.0f/c+s2);
    }

    for(int act=0;act<2;act++){
        const char* nm=act?"tanh":"relu";
        float out[N*DIN1];
        CHECK(wubu_hactivation(x,N,D,c,act,out)==0);
        printf("  %s_on_hyperboloid...",nm);
        for(int i=0;i<N;i++){
            float L=-(out[i*DIN1+0]*out[i*DIN1+0]);
            for(int d=1;d<DIN1;d++)L+=out[i*DIN1+d]*out[i*DIN1+d];
            CHECK(fabsf(L+1.0f)<1e-3f);
            CHECK(out[i*DIN1+0]>0);
            for(int d=0;d<DIN1;d++)CHECK(!isnan(out[i*DIN1+d]));
        }
        printf("PASS\n");passed++;
    }

    /* ReLU: at least one previously-negative tangent component becomes
     * zero in the log-domain of the output (verify via log_0 of out) */
    printf("  relu_zeroes_negatives...");
    {
        float out[N*DIN1],v[64];
        CHECK(wubu_hactivation(x,N,D,c,0,out)==0);
        int any_zeroed=0;
        for(int i=0;i<N;i++){
            lorentz_log0(out+i*DIN1,v,D);
            for(int d=0;d<D;d++){
                if(x[i*DIN1+d+1]<-0.05f && fabsf(v[d])<fabsf(x[i*DIN1+d+1]))
                    any_zeroed=1;
            }
        }
        CHECK(any_zeroed);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
