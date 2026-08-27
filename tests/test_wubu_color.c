#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_color.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Color Space Tests ===\n\n");

    printf("  g1_601_vs_709_green_weight...");
    {
        uint8_t rgb[3]={0,255,0};
        uint8_t y601,y709,cb,cr;
        wubu_rgb_to_yuv_cm(rgb,&y601,&cb,&cr,1,WUBU_CS_BT601);
        wubu_rgb_to_yuv_cm(rgb,&y709,&cb,&cr,1,WUBU_CS_BT709);
        printf("[601_Y=%d 709_Y=%d] ",y601,y709);
        CHECK(y601<150&&y709>175);
    }
    printf("PASS\n");passed++;

    printf("  g2_tonemap...");
    {
        float hdr[30];
        uint8_t sdr[30];
        for(int i=0;i<10;i++){
            hdr[i*3]=(float)(i*1000);
            hdr[i*3+1]=(float)(i*1000);
            hdr[i*3+2]=(float)(i*1000);
        }
        wubu_tonemap_reinhard(hdr,sdr,10,10000.0f);
        CHECK(sdr[0]<50&&sdr[9]>200);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
