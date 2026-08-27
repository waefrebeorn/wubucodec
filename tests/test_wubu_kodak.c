/* test_wubu_kodak.c -- GAP-E003 gates: audio -> image -> audio round trip */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_kodak.h"
#include "wubu_stft.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
int main(void){
    printf("=== WuBuMath Kodak Tests ===\n\n");
    const int LEN=8192,F=512,HOP=128;
    WubuStft st;CHECK(wubu_stft_init(&st,F,HOP)==0);
    int T=wubu_stft_num_frames(LEN,F,HOP);

    /* audio: moving tone + harmonic + noise */
    float x[LEN];
    unsigned r=3u;
    for(int i=0;i<LEN;i++){
        r=r*1103515245u+12345u;
        float n=(float)(r>>16)/65536.0f-0.5f;
        float sweep=200.0f+300.0f*(float)i/LEN;
        x[i]=0.5f*sinf(2*(float)M_PI*sweep*(float)i/48000.0f)
            +0.3f*sinf(2*(float)M_PI*2*sweep*(float)i/48000.0f+1.0f)
            +0.2f*n;
    }

    int bins=F/2+1;
    float* S=wubu_stft_forward(&st,x,LEN,&T);
    CHECK(S&&T>0);

    /* pack into 256x256 image */
    WubuKodak kd;CHECK(wubu_kodak_init(&kd,256,256)==0);
    CHECK(wubu_kodak_pack(&kd,S,T,bins)==0);
    printf("  pack_image...PASS\n");passed++;

    /* unpack and invert */
    float* S2=malloc(sizeof(float)*(size_t)T*bins*2);
    CHECK(wubu_kodak_unpack(&kd,S2,T,bins)==0);
    float* y=wubu_stft_inverse(&st,S2,T,LEN);
    CHECK(y);

    double num=0,dx=0,dy=0,mxerr=0;
    for(int i=512;i<LEN-512;i++){
        num+=(double)x[i]*y[i];
        dx+=(double)x[i]*x[i];dy+=(double)y[i]*(double)y[i];
        float e=fabsf(x[i]-y[i]);if(e>mxerr)mxerr=e;
    }
    float corr=(float)(num/(sqrt(dx*dy)+1e-12));
    printf("  round_trip...  [corr=%.4f] ",(double)corr);
    CHECK(corr>0.90f);   /* lossy layout (bin resampling), still faithful */
    free(S);free(S2);free(y);wubu_kodak_free(&kd);wubu_stft_free(&st);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
