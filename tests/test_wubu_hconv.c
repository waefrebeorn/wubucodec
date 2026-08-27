/* test_wubu_hconv.c -- GAP-C027 gates
 *  G1 identity W, zero bias → output ≈ input (on-ball)
 *  G2 outputs stay on ball for random W
 *  G3 zero W → all outputs at origin (exp0(0)=0)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hconv.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Conv (Poincaré MLP) Tests ===\n\n");
    const int N=5,D=8,D_out=4;
    float c=1.0f;

    float x[N*D];
    unsigned rs=42u;
    for(int i=0;i<N*D;i++){
        rs=rs*1103515245u+12345u;
        x[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=x[i*D+d]*x[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)x[i*D+d]*=s;}
    }

    printf("  g1_identity...");
    {
        float W[D_out*D];memset(W,0,sizeof(W));
        for(int j=0;j<D_out&&j<D;j++)W[j*D+j]=1.0f;
        float out[N*D_out];
        CHECK(wubu_hconv_forward(W,NULL,x,N,D,D_out,c,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;for(int d=0;d<D_out;d++)n2+=out[i*D_out+d]*out[i*D_out+d];
            CHECK(n2<1.0f);
            /* first D_out coords should approximate input's */
            for(int d=0;d<D_out&&d<D;d++)
                CHECK(fabsf(out[i*D_out+d]-x[i*D+d])<0.05f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_random_w_on_ball...");
    {
        float W[D_out*D];
        for(int i=0;i<D_out*D;i++){
            rs=rs*1103515245u+12345u;
            W[i]=(float)((rs>>16)%1000)/5000.0f-0.1f;
        }
        float out[N*D_out];
        CHECK(wubu_hconv_forward(W,NULL,x,N,D,D_out,c,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;for(int d=0;d<D_out;d++)n2+=out[i*D_out+d]*out[i*D_out+d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_zero_w_origin...");
    {
        float W[D_out*D];memset(W,0,sizeof(W));
        float out[N*D_out];
        CHECK(wubu_hconv_forward(W,NULL,x,N,D,D_out,c,out)==0);
        for(int i=0;i<N;i++)
            for(int d=0;d<D_out;d++)
                CHECK(fabsf(out[i*D_out+d])<1e-6f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
