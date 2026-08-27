#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_dpb.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== DPB + Weighted Pred Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_dpb_fifo...");
    {
        DPB* dpb=wubu_dpb_create(4,W,H);
        /* push 6 frames, verify oldest are evicted */
        for(int i=0;i<6;i++){
            uint8_t* f=calloc((size_t)W*H*3,1);
            memset(f,(uint8_t)(i*40),W*(size_t)H*3);
            wubu_dpb_push(dpb,f,i,W,H);
            free(f);
        }
        CHECK(wubu_dpb_count(dpb)==4);  /* capped at max_frames */
        CHECK(wubu_dpb_get_poc(dpb,0)==5);  /* most recent */
        CHECK(wubu_dpb_get_poc(dpb,3)==2);  /* oldest surviving */
        wubu_dpb_destroy(dpb);
    }
    printf("PASS\n");passed++;

    printf("  g2_weighted_pred...");
    {
        long n=100;
        uint8_t* r0=malloc((size_t)n);
        uint8_t* r1=malloc((size_t)n);
        uint8_t* out=malloc((size_t)n);
        memset(r0,100,(size_t)n);
        memset(r1,200,(size_t)n);
        
        /* equal weights → average = 150 */
        wubu_weighted_pred(r0,r1,out,n,1,1,0,1);
        CHECK(out[50]==150);
        
        /* all weight on ref0 → 100 */
        wubu_weighted_pred(r0,NULL,out,n,2,0,0,1);
        CHECK(out[50]==100);
        
        free(r0);free(r1);free(out);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
