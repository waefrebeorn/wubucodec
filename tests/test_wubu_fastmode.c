#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_fastmode.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Fast Mode Decision Tests ===\n\n");

    printf("  g1_early_skip_identical...");
    {
        /* identical prediction → skip immediately */
        CHECK(wubu_fm_early_skip(0,10,100)==1);
        /* high SAD → cannot skip */
        CHECK(wubu_fm_early_skip(500,10,100)==0);
    }
    printf("PASS\n");passed++;

    printf("  g2_fm_decision_stages...");
    {
        /* stage 1: skip */
        CHECK(wubu_fm_decide(5,10,100,1000)==FM_DECISION_SKIP);
        /* stage 2: inter only */
        CHECK(wubu_fm_decide(500,10,100,1000)==FM_DECISION_INTER_ONLY);
        /* stage 3: full search */
        CHECK(wubu_fm_decide(2000,10,100,1000)==FM_DECISION_FULL_SEARCH);
    }
    printf("PASS\n");passed++;

    printf("  g3_aq_detail_detection...");
    {
        const int W=32,H=32;
        uint8_t* img=calloc((size_t)W*H,1);
        
        /* flat block → positive offset (raise QP) */
        memset(img,128,(size_t)W*H);
        int offset_flat=wubu_aq_offset(img,W,H,4,4,8,4);
        
        /* noisy block → negative offset (lower QP) */
        srand(42);
        for(long i=0;i<(long)W*H;i++)img[i]=rand()%256;
        int offset_noisy=wubu_aq_offset(img,W,H,4,4,8,4);
        
        printf("[flat=%d noisy=%d] ",offset_flat,offset_noisy);
        CHECK(offset_flat>offset_noisy); /* flat gets higher QP than detailed */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
