/* test_wubu_ldirect.c -- GAP-C022 gates: tangent-space-free Lorentz linear
 *  G1 output stays on the hyperboloid L(x,x) = -1/c, upper sheet
 *  G2 identity W reproduces input (time recomputed from same space)
 *  G3 zero W: output is exp_0(b) equivalent (all space=b, time from Eq.3)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_ldirect.h"
#include "wubu_lorentz.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Tangent-Free Lorentz Linear Tests ===\n\n");
    const int D_in=8,D_out=4,N=3;
    const int DIN1=D_in+1,DOUT1=D_out+1;
    float c=1.0f;

    /* points on the hyperboloid */
    float x[N*DIN1];
    for(int i=0;i<N;i++){
        for(int d=1;d<DIN1;d++)
            x[i*DIN1+d]=((float)((i*13+d*29)%89)/89.0f-0.5f)*0.6f;
        float s2=0;for(int d=1;d<DIN1;d++)s2+=x[i*DIN1+d]*x[i*DIN1+d];
        x[i*DIN1+0]=sqrtf(1.0f/c+s2);
    }

    /* G1: identity W — output stays on hyperboloid */
    printf("  g1_identity_on_hyperboloid...");
    {
        float W[D_out*D_in];memset(W,0,sizeof(W));
        for(int j=0;j<D_out&&j<D_in;j++)W[j*D_in+j]=1.0f;
        float out[N*DOUT1];
        CHECK(wubu_ldirect_forward(W,NULL,x,N,D_in,D_out,c,out)==0);
        for(int i=0;i<N;i++){
            float L=-(out[i*DOUT1+0]*out[i*DOUT1+0]);
            for(int d=1;d<DOUT1;d++)L+=out[i*DOUT1+d]*out[i*DOUT1+d];
            CHECK(fabsf(L+1.0f)<1e-4f);   /* L = -1/c = -1 */
            CHECK(out[i*DOUT1+0]>0);
        }
    }
    printf("PASS\n");passed++;

    /* G2: identity W + zero bias reproduces input's first D_out dims */
    printf("  g2_identity_reproduces...");
    {
        float W[D_out*D_in];memset(W,0,sizeof(W));
        for(int j=0;j<D_out&&j<D_in;j++)W[j*D_in+j]=1.0f;
        float out[N*DOUT1];
        CHECK(wubu_ldirect_forward(W,NULL,x,N,D_in,D_out,c,out)==0);
        for(int i=0;i<N;i++)
            for(int d=1;d<=D_out;d++)
                CHECK(fabsf(out[i*DOUT1+d]-x[i*DIN1+d])<1e-5f);
    }
    printf("PASS\n");passed++;

    /* G3: zero W → all outputs equal exp_0(b) equivalent */
    printf("  g3_zero_w_bias_dominates...");
    {
        float W[D_out*D_in];memset(W,0,sizeof(W));
        float b[D_out];for(int d=0;d<D_out;d++)b[d]=0.05f;
        float out[N*DOUT1];
        CHECK(wubu_ldirect_forward(W,b,x,N,D_in,D_out,c,out)==0);
        /* expected time from Eq.3: sqrt(1/c + |b|²) */
        float bn2=0;for(int d=0;d<D_out;d++)bn2+=b[d]*b[d];
        float expected_t=sqrtf(1.0f/c+bn2);
        for(int i=0;i<N;i++){
            CHECK(fabsf(out[i*DOUT1+0]-expected_t)<1e-5f);
            for(int d=1;d<=D_out;d++)
                CHECK(fabsf(out[i*DOUT1+d]-b[d-1])<1e-6f);
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
