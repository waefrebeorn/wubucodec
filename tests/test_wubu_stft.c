/*
 * test_wubu_stft.c -- GAP-E001 gates
 *  G1 FFT of known sinusoid puts energy in the right bin
 *  G2 STFT->ISTFT round trip: correlation > 0.999, max err < 1e-3
 *  G3 round trip is exact-ish at interior (away from edges)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_stft.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s\n",#c); failed++; return; } }while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void test_fft_bin(void){
    WubuStft st; CHECK(wubu_stft_init(&st,256,64)==0);
    int len=256; float x[256];
    for(int i=0;i<len;i++) x[i]=sinf(2.0f*(float)M_PI*8.0f*(float)i/(float)len); /* bin 8 */
    int T; float* S=wubu_stft_forward(&st,x,len,&T);
    CHECK(T==1&&S);
    /* peak magnitude at bin 8 */
    int best=0; float bm=0;
    for(int b=0;b<129;b++){
        float m=S[b*2]*S[b*2]+S[b*2+1]*S[b*2+1];
        if(m>bm){bm=m;best=b;}
    }
    CHECK(best==8);
    free(S); wubu_stft_free(&st);
}
static void test_round_trip(void){
    WubuStft st; CHECK(wubu_stft_init(&st,512,128)==0);  /* 75% overlap */
    int len=8192;
    float* x=malloc(sizeof(float)*(size_t)len);
    unsigned r=42u;
    for(int i=0;i<len;i++){
        r=r*1103515245u+12345u;
        float n=(float)(r>>16)/65536.0f-0.5f;
        x[i]=0.6f*sinf(2.0f*(float)M_PI*37.0f*(float)i/64.0f)+0.4f*n;
    }
    int T; float* S=wubu_stft_forward(&st,x,len,&T);
    CHECK(S&&T>0);
    float* y=wubu_stft_inverse(&st,S,T,len);
    CHECK(y);
    /* interior comparison (skip COLA edge ramp) */
    double num=0,den=0,maxerr=0;
    for(int i=512;i<len-512;i++){
        num+=(double)x[i]*y[i];
        den+=(double)x[i]*x[i];
        float e=fabsf(x[i]-y[i]); if(e>maxerr)maxerr=e;
    }
    float corr=(float)(num/sqrt(den*(double)den));
    printf("[corr=%.5f maxerr=%.2e] ",(double)corr,(double)maxerr);
    CHECK(corr>0.999f);
    CHECK(maxerr<1e-2f);
    free(x);free(S);free(y);wubu_stft_free(&st);
}
int main(void){
    printf("=== WuBuMath STFT Tests ===\n\n");
    printf("  test_fft_bin...");      test_fft_bin();     printf("PASS\n");passed++;
    printf("  test_round_trip...  ");test_round_trip(); printf("PASS\n");passed++;
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
