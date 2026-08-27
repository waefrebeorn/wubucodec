#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_deblock.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Deblocking Filter Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_smooths_block_boundary...");
    {
        /* create two flat regions with a step at x=32 (8-block boundary) */
        uint8_t* img=malloc((size_t)W*H*3);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                int v=(x<32)?100:110;  /* small step = artifact */
                for(int c=0;c<3;c++)img[((size_t)y*W+x)*3+c]=(uint8_t)v;
            }
        wubu_db_filter(img,W,H,5);
        /* after filtering, boundary pixels should be closer together */
        int before_diff=10,after_diff=0;
        for(int y=0;y<H;y++){
            int d=img[((size_t)y*W+31)*3]-img[((size_t)y*W+32)*3];
            if(d<0)d=-d;
            after_diff+=d;
        }
        printf("[boundary diff after=%d] ",after_diff);
        CHECK(after_diff<before_diff*(int)H);  /* reduced */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g2_preserves_real_edges...");
    {
        uint8_t* img=malloc((size_t)W*H*3);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                int v=(x<32)?30:220;  /* LARGE step = real edge */
                for(int c=0;c<3;c++)img[((size_t)y*W+x)*3+c]=(uint8_t)v;
            }
        wubu_db_filter(img,W,H,5);
        /* edge should still exist */
        int diff=img[((size_t)H/2*W+31)*3]-img[((size_t)H/2*W+32)*3];
        if(diff<0)diff=-diff;
        printf("[edge diff=%d] ",diff);
        CHECK(diff>100);  /* real edge preserved */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
