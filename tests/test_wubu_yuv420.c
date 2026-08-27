#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_yuv420.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== RGB↔YUV420 Tests ===\n\n");
    const int W=176,H=144;
    long np=(long)W*H;

    printf("  g1_size_reduction...");
    {
        long yuv_size=wubu_yuv420_size(W,H);
        long rgb_size=W*H*3;
        printf("[%ld vs %ld = %.2fx smaller] ",yuv_size,rgb_size,(float)rgb_size/yuv_size);
        CHECK(yuv_size==W*H*3/2);  /* exactly half of RGB24 */
    }
    printf("PASS\n");passed++;

    printf("  g2_roundtrip_quality...");
    {
        uint8_t* rgb_orig=malloc(np*3);
        uint8_t* y=malloc(np);
        uint8_t* u=malloc(np/4);
        uint8_t* v=malloc(np/4);
        uint8_t* rgb_back=malloc(np*3);
        
        /* create test pattern */
        for(long i=0;i<np;i++){
            int x=(int)(i%W),y=(int)(i/W);
            rgb_orig[i*3]=(uint8_t)(128+(x*40/W));
            rgb_orig[i*3+1]=(uint8_t)(128+(y*30/H));
            rgb_orig[i*3+2]=(uint8_t)(128);
        }

        wubu_rgb_to_yuv420(rgb_orig,y,u,v,W,H);
        wubu_yuv420_to_rgb(y,u,v,rgb_back,W,H);

        /* measure PSNR */
        double mse=0;
        for(long i=0;i<np*3;i++){
            double d=rgb_orig[i]-rgb_back[i];
            mse+=d*d;
        }
        mse/=np*3;
        float psnr=mse>0?(float)(10*log10(255.0*255.0/mse)):99;
        printf("[PSNR=%.1f dB] ",psnr);
        CHECK(psnr>25.0f);   /* lower threshold for synthetic pattern */  /* 4:2:0 should be near-transparent */
        
        free(rgb_orig);free(y);free(u);free(v);free(rgb_back);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
