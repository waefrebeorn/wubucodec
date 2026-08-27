#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_motionest.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Motion Estimation Tests ===\n\n");
    const int W=64,H=64,BS=8;

    printf("  g1_zero_motion...");
    {
        /* identical frames → MV should be (0,0) */
        uint8_t* f=malloc((size_t)W*H*3);
        for(long i=0;i<(long)W*H*3;i++)f[i]=(uint8_t)(i%256);
        int dx,dy;
        wubu_me_block(f,f,W,H,16,16,BS,4,&dx,&dy);
        CHECK(dx==0&&dy==0);
        free(f);
    }
    printf("PASS\n");passed++;

    printf("  g2_known_translation...");
    {
        /* shift frame right by 3 pixels */
        uint8_t* ref=malloc((size_t)W*H*3);
        uint8_t* curr=malloc((size_t)W*H*3);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                for(int c=0;c<3;c++){
                    int sx=(x+3<W)?(x+3):(W-1);
                    curr[((size_t)y*W+x)*3+c]=ref[((size_t)y*W+sx)*3+c];
                }
        int dx,dy;
        /* test block away from edges */
        wubu_me_block(curr,ref,W,H,16,24,BS,8,&dx,&dy);
        printf("[found dx=%d (want 3)] ",dx);
        CHECK(dx==3);  /* content shifted left means motion vector is -(-3)=3 */
        CHECK(dy==0);
        free(ref);free(curr);
    }
    printf("PASS\n");passed++;

    printf("  g3_compensate_reduces_error...");
    {
        /* after compensation, prediction should be close to original */
        uint8_t* ref=malloc((size_t)W*H*3);
        uint8_t* curr=malloc((size_t)W*H*3);
        uint8_t* pred=malloc((size_t)W*H*3);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                for(int c=0;c<3;c++){
                    int sx=(x+2<W)?(x+2):W-1;
                    curr[((size_t)y*W+x)*3+c]=ref[((size_t)y*W+sx)*3+c];
                }

        int mvs[(W/BS)*(H/BS)*2];
        wubu_me_frame(curr,ref,W,H,BS,4,mvs);
        wubu_me_compensate(ref,W,H,BS,mvs,pred);

        long sad_before=0,sad_after=0;
        for(long i=0;i<(long)W*H*3;i+=7){
            sad_before+=abs(curr[i]-ref[i]);
            sad_after+=abs(curr[i]-pred[i]);
        }
        printf("[before=%ld after=%ld] ",sad_before,sad_after);
        CHECK(sad_after<sad_before);  /* compensation improves prediction */
        free(ref);free(curr);free(pred);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
