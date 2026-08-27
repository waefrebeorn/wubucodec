#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_partitions.h"
#include "wubu_motionest.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Partition + Search Tests ===\n\n");
    const int W=64,H=64,BS=8;

    printf("  g1_block_variance...");
    {
        uint8_t* img=calloc((size_t)W*H,1);
        /* flat area → variance=0 */
        memset(img,128,(size_t)W*H);
        CHECK(wubu_block_variance(img,W,H,16,16,BS)==0);
        
        /* high contrast → high variance */
        for(int i=0;i<W*(size_t)H;i+=2)img[i]=255;
        CHECK(wubu_block_variance(img,W,H,16,16,BS)>1000);
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g2_partitions_16x16...");
    {
        wubu_partition_t parts[10];
        int n=wubu_enum_partitions(16,parts);
        printf("[%d partitions] ",n);
        CHECK(n>=5); /* at least 16x16 + 2×16x8 + 2×8x16 = 5 */
        /* verify 16x16 is first */
        CHECK(parts[0].type==PART_16x16&&parts[0].w==16);
    }
    printf("PASS\n");passed++;

    printf("  g3_diamond_finds_match...");
    {
        uint8_t* ref=malloc((size_t)W*H);
        uint8_t* curr=malloc((size_t)W*H);
        /* textured pattern shifted by (3,-1) */
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                ref[(size_t)y*W+x]=(uint8_t)((x*x/7+y*y/5+x*y/3)%256);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                int sx=x-3,sy=y+1;
                if(sx<0)sx=0;if(sx>=W)sx=W-1;
                if(sy<0)sy=0;if(sy>=H)sy=H-1;
                curr[(size_t)y*W+x]=ref[(size_t)sy*W+sx];
            }
        
        int dx,dy;
        long sad=wubu_diamond_search(curr,ref,W,H,16,16,BS,8,&dx,&dy);
        printf("[MV=(%d,%d) SAD=%ld] ",dx,dy,sad);
        CHECK(abs(dx-(-3))<=1);  /* should find approximately (-3,+1) */
        free(ref);free(curr);
    }
    printf("PASS\n");passed++;

    printf("  g4_hexagon_finds_large_motion...");
    {
        uint8_t* ref=malloc((size_t)W*H);
        uint8_t* curr=malloc((size_t)W*H);
        /* shift by 6 pixels — needs wider search than diamond default */
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                ref[(size_t)y*W+x]=(uint8_t)((x*x/7+y*y/5+x*y/3)%256);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                int sx=x-6;if(sx<0)sx=0;if(sx>=W)sx=W-1;
                curr[(size_t)y*W+x]=ref[(size_t)y*W+sx];
            }
        
        int dx,dy;
        long sad=wubu_hexagon_search(curr,ref,W,H,16,16,BS,8,&dx,&dy);
        printf("[MV=(%d,%d) SAD=%ld] ",dx,dy,sad);
        CHECK(abs(dx-(-6))<=2);  /* should find approximately -6 */
        free(ref);free(curr);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
