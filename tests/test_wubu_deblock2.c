#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_deblock2.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== H.264 Deblocking Tests ===\n\n");

    printf("  g1_preserves_real_edges...");
    {
        /* large step = real edge → should NOT be smoothed at high QP */
        uint8_t* img=calloc(32*32,1);
        for(int y=0;y<32;y++)
            for(int x=0;x<32;x++)
                img[(size_t)y*32+x]=(x<16)?30:220; /* huge step */
        
        wubu_deblock_h264(img,32,32,10,3); /* low QP = gentle filtering */
        
        int diff=img[(size_t)15*32+14]-img[(size_t)15*32+17];
        if(diff<0)diff=-diff;
        printf("[edge_diff=%d] ",diff);
        CHECK(diff>100); /* real edge preserved */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g2_smooths_small_artifact...");
    {
        /* small step = quantization artifact → SHOULD be smoothed */
        uint8_t* img=calloc(32*32,1);
        for(int y=0;y<32;y++)
            for(int x=0;x<32;x++)
                img[(size_t)y*32+x]=(x<16)?120:128; /* 8-level step */
        
        wubu_deblock_h264(img,32,32,30,3);
        
        int d=img[(size_t)15*32+15]-img[(size_t)15*32+16];
        if(d<0)d=-d;
        printf("[boundary_diff_after=%d (was 8)] ",d);
        CHECK(d<8); /* boundary pixels smoothed */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
