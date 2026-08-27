#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_aud2img.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Audio→Image Codec Tests ===\n\n");
    const int NS=1024,FFT=64,HOP=16,IW=64,IH=33;
    float audio[NS],back[NS];
    unsigned rs=42u;
    /* sine sweep + harmonics */
    for(int i=0;i<NS;i++){
        float t=(float)i/NS*2*3.14159265f*5;
        back[i]=sinf(t)+0.3f*sinf(2*t);
    }
    uint8_t img[(size_t)IW*IH*3];

    printf("  g1_encode_produces_image...");
    {
        int nf=wubu_ai_encode(audio,NS,FFT,HOP,IW,IH,img);
        CHECK(nf>0);
        /* image should be non-zero */
        int nonzero=0;
        for(long i=0;i<(long)IW*IH*3;i++)if(img[i])nonzero++;
        CHECK(nonzero>100);
    }
    printf("PASS\n");passed++;
    printf("  g2_decode_correlation...");
    {
        memset(img,0,sizeof(img));
        wubu_ai_encode(audio,NS,FFT,HOP,IW,IH,img);
        wubu_ai_decode(img,IW,IH,FFT,HOP,back,NS);
        float corr=wubu_ai_correlation(audio,back,NS);
        printf("[corr=%.4f] ",corr);
        CHECK(corr>0.15f);   /* positive correlation = signal survived */
    }
    printf("PASS\n");passed++;
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
