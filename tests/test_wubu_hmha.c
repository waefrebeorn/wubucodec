/* test_wubu_hmha.c -- GAP-C033 gates
 *  G1 outputs on-ball
 *  G2 all finite
 *  G3 deterministic
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hmha.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Multi-Head Attention Tests ===\n\n");
    const int N=4,D=8,H=2;
    float c=1.0f,tau=0.5f;

    float x[N*D],Wq[D*D],Wk[D*D],Wv[D*D],Wo[D*D];
    unsigned rs=42u;
    for(int i=0;i<N*D;i++){
        rs=rs*1103515245u+12345u;
        x[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=x[i*D+d]*x[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)x[i*D+d]*=s;}
    }
    for(int i=0;i<D*D;i++){
        rs=rs*1103515245u+12345u;
        Wq[i]=Wk[i]=Wv[i]=Wo[i]=(float)((rs>>16)%1000)/25000.0f-0.02f;
    }

    float out[N*D];
    int rc=wubu_hmha_forward(Wq,Wk,Wv,Wo,x,N,D,H,tau,c,out);
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
    for(int i=0;i<N*D;i++){
        CHECK(!isnan(out[i]));
        CHECK(!isinf(out[i]));
    }
    printf("PASS\n");passed++;

    printf("  g3_deterministic...");
    {
        float out2[N*D];
        CHECK(wubu_hmha_forward(Wq,Wk,Wv,Wo,x,N,D,H,tau,c,out2)==0);
        for(int i=0;i<N*D;i++)CHECK(fabsf(out[i]-out2[i])<1e-7f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
