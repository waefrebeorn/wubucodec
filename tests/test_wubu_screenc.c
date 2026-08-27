#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_screenc.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Screen Content Coding Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_ibc_finds_repeated_block...");
    {
        /* create image with a repeated pattern */
        uint8_t* img=calloc((size_t)W*H,1);
        for(int r=0;r<8;r++)for(int c=0;c<8;c++){
            img[(size_t)(4+r)*W+(4+c)]=(uint8_t)((r*c+1)%256);  /* block at (4,4) */
            img[(size_t)(20+r)*W+(20+c)]=img[(size_t)(4+r)*W+(4+c)];  /* same at (20,20) */
        }
        
        int found_bx,found_by;
        long sad=wubu_ibc_search(img,W,H,20,20,8,32,&found_bx,&found_by);
        
        printf("[found=(%d,%d) SAD=%ld] ",found_bx,found_by,sad);
        CHECK(found_bx==4&&found_by==4); /* should find the original */
        CHECK(sad==0); /* exact match */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g2_palette_roundtrip...");
    {
        uint8_t* img=malloc((size_t)W*H);
        /* create block with only 4 unique colors */
        uint8_t colors[4]={10,50,100,200};
        for(int r=0;r<8;r++)
            for(int c=0;c<8;c++)
                img[(size_t)(8+r)*W+(8+c)]=colors[(r+c)%4];
        
        uint8_t palette[16];
        int n=wubu_palette_extract(img,W,H,8,8,8,palette,16);
        CHECK(n==4); /* exactly 4 colors */
        
        uint8_t indices[64],decoded[64];
        int err=wubu_palette_encode(img,W,H,8,8,8,palette,n,indices);
        CHECK(err==0);
        
        wubu_palette_decode(indices,palette,n,decoded,64);
        /* verify round trip */
        for(int i=0;i<64;i++)
            CHECK(decoded[i]==img[(size_t)(8+i/8)*W+(8+i%8)]);
        free(img);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
