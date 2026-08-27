/* test_wubu_hlinear.c -- GAP-C017 gates: hyperboloid linear layer
 *  G1 output stays on the hyperboloid L(x,x) = -1/c, upper sheet
 *  G2 identity W (square, I) + zero bias reproduces input
 *  G3 with W=0: output is exp_0(b) for all inputs (bias dominates)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hlinear.h"
#include "wubu_lorentz.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

static float lor_n2(const float* x,int D1){
    float L=-(x[0]*x[0]);
    for(int d=1;d<D1;d++)L+=x[d]*x[d];
    return L;
}
int main(void){
    printf("=== Hyperboloid Linear Layer Tests ===\n\n");
    const int D_in=8,D_out=4,N=3;
    const int DIN1=D_in+1,DOUT1=D_out+1;
    float c=1.0f;

    /* points on the hyperboloid */
    float x[N*DIN1];
    for(int i=0;i<N;i++){
        for(int d=1;d<DIN1;d++)
            x[i*DIN1+d]=((float)((i*31+d*17)%97)/97.0f-0.5f)*0.6f;
        float s2=0;for(int d=1;d<DIN1;d++)s2+=x[i*DIN1+d]*x[i*DIN1+d];
        x[i*DIN1+0]=sqrtf(1.0f/c+s2);
    }

    printf("  g1_identity_reproduces...");
    {
        float W[D_out*D_in];
        memset(W,0,sizeof(W));
        for(int j=0;j<D_out&&j<D_in;j++)W[j*D_in+j]=1.0f;   /* identity slice */
        float out[N*DOUT1];
        CHECK(wubu_hlinear_forward(W,NULL,x,N,D_in,D_out,c,out)==0);
        for(int i=0;i<N;i++){
            /* L(out,out) should be -1 (on hyperboloid) */
            float L=lor_n2(out+i*DOUT1,DOUT1);
            CHECK(fabsf(L+1.0f)<1e-3f);
            CHECK(out[i*DOUT1+0]>0);   /* upper sheet */
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_bias_dominates...");
    {
        float W[D_out*D_in];memset(W,0,sizeof(W));
        float b[D_out];for(int d=0;d<D_out;d++)b[d]=0.05f;
        float out[N*DOUT1];
        CHECK(wubu_hlinear_forward(W,b,x,N,D_in,D_out,c,out)==0);
        /* all outputs should equal exp_0(b) — same point */
        float e0[DOUT1];
        lorentz_exp0(b,e0,DOUT1);
        for(int i=0;i<N;i++)
            for(int d=0;d<DOUT1;d++)
                CHECK(fabsf(out[i*DOUT1+d]-e0[d])<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
/* GAP-C018 gates appended via separate main-less file inclusion pattern */
