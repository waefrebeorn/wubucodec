/* test_wubu_hblock.c -- GAP-C032 gates
 *  G1 outputs stay on-ball through the FULL block
 *  G2 all outputs finite (no NaN from chained ops)
 *  G3 identity weights: output finite and on-ball (residual path works)
 *  G4 deterministic across repeated calls
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hblock.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

static void init_ball(float* x,int n,int D){
    unsigned rs=42u;
    for(int i=0;i<n*D;i++){
        rs=rs*1103515245u+12345u;
        x[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    for(int i=0;i<n;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=x[i*D+d]*x[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)x[i*D+d]*=s;}
    }
}
int main(void){
    printf("=== Hyperbolic Transformer Block Tests ===\n\n");
    const int N=4,D=16;
    float c=1.0f;
    float x[N*D];
    init_ball(x,N,D);

    /* small random weights */
    float W_att[D*D],W_ff1[D*D],W_ff2[D*D];
    float b_att[16]={0},b_ff1[16]={0},b_ff2[16]={0};
    unsigned rs=7u;
    for(int i=0;i<D*D;i++){
        rs=rs*1103515245u+12345u;
        float v=(float)((rs>>16)%1000)/50000.0f-0.01f;
        W_att[i]=v;W_ff1[i]=v;W_ff2[i]=v*0.5f;
    }

    float out[N*D];
    int rc=wubu_hblock_forward(W_att,b_att,W_ff1,b_ff1,W_ff2,b_ff2,
                                NULL,NULL,x,N,D,c,out);
    CHECK(rc==0);

    printf("  g1_on_ball...");
    {
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=out[i*D+d]*out[i*D+d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_finite...");
    {
        for(int i=0;i<N*D;i++){
            CHECK(!isnan(out[i]));
            CHECK(!isinf(out[i]));
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_identity_weights...");
    {
        /* zero all weights → block reduces to norms + mobius_add(0) = norm */
        float W0[D*D];memset(W0,0,sizeof(W0));
        float out0[N*D];
        CHECK(wubu_hblock_forward(W0,NULL,W0,NULL,W0,NULL,
                                    NULL,NULL,x,N,D,c,out0)==0);
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=out0[i*D+d]*out0[i*D+d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g4_deterministic...");
    {
        float out2[N*D];
        CHECK(wubu_hblock_forward(W_att,b_att,W_ff1,b_ff1,W_ff2,b_ff2,
                                    NULL,NULL,x,N,D,c,out2)==0);
        for(int i=0;i<N*D;i++)CHECK(fabsf(out[i]-out2[i])<1e-7f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
