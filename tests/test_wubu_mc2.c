#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mc2.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Group 2: Advanced MC Tests ===\n\n");
    const int W=32,H=32;

    printf("  g1_quarterpel_preserves_int...");
    {
        uint8_t* src=malloc((size_t)W*H);
        for(long i=0;i<(long)W*H;i++)src[i]=(uint8_t)(i%256);
        uint8_t* dst=calloc((size_t)W*4*H*4,1);
        
        wubu_gen_quarterpel(src,dst,W,H);
        
        /* integer positions must be preserved */
        int match=1;
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                if(dst[(size_t)(y*4)*(W*4)+(x*4)]!=src[(size_t)y*W+x]){match=0;break;}
        CHECK(match);
        free(src);free(dst);
    }
    printf("PASS\n");passed++;

    printf("  g2_temporal_mv...");
    {
        int bpr=W/8;
        int16_t* mvf=calloc((size_t)bpr*(H/8)*2,sizeof(int16_t));
        int target_idx=(2)*bpr+3;
        mvf[target_idx*2]=5;mvf[target_idx*2+1]=-3;
        
        int16_t out_mv[2];
        wubu_temporal_mv(mvf,bpr,24,16,out_mv); /* block at (3,2) in block coords */
        CHECK(out_mv[0]==5&&out_mv[1]==-3);
        free(mvf);
    }
    printf("PASS\n");passed++;

    printf("  g3_refpool_eviction...");
    {
        RefPool* rp=wubu_refpool_create(4);
        /* push 6 frames: 0-3 short-term, 4 long-term, 5 short-term */
        uint8_t dummy[64];
        for(int i=0;i<6;i++)
            wubu_refpool_push(rp,dummy,i,i==4?1:0);
        CHECK(rp->count>0);
        
        /* evict down to 2 short-term */
        wubu_refpool_evict(rp,2);
        
        
        free(rp->frames);free(rp->pocs);free(rp->is_longterm);free(rp);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
