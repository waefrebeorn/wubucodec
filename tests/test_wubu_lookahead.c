#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_lookahead.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Lookahead + VBV Tests ===\n\n");

    printf("  g1_vbv_underflow_prevention...");
    {
        /* small buffer, high bitrate frame → should prevent decode */
        WubuVbv vbv;
        wubu_vbv_init(&vbv,10000,500,30); /* 10Kbit buffer, 500bps */
        
        CHECK(wubu_vbv_can_decode(&vbv,20000)==0); /* too big */
        
        long max=wubu_vbv_max_frame_bits(&vbv);
        printf("[max_frame=%ld] ",max);
        CHECK(max>0&&max<=12000); /* reasonable limit */
    }
    printf("PASS\n");passed++;

    printf("  g2_la_detects_scenecut...");
    {
        const int W=64,H=64;
        Lookahead* la=wubu_la_create(5,W,H);
        
        /* push dark frame then bright frame (scene cut) */
        uint8_t* dark=calloc((size_t)W*H,1);
        uint8_t* bright=malloc((size_t)W*H);
        memset(bright,200,(size_t)W*H);
        
        wubu_la_push(la,dark,W,H);
        wubu_la_push(la,bright,W,H);
        
        CHECK(wubu_la_has_scenecut(la)==1);
        wubu_la_destroy(la);
        free(dark);free(bright);
    }
    printf("PASS\n");passed++;

    printf("  g3_qp_adjustment...");
    {
        const int W=64,H=64;
        Lookahead* la=wubu_la_create(3,W,H);
        
        uint8_t* f1=calloc((size_t)W*H,1);
        uint8_t* f2=malloc((size_t)W*H);
        memset(f2,200,(size_t)W*H);
        wubu_la_push(la,f1,W,H);
        wubu_la_push(la,f2,W,H); /* scene change! */
        
        int qp=wubu_la_adjust_qp(la,23,1.0);
        printf("[adjusted_qp=%d] ",qp);
        CHECK(qp<23); /* should lower QP for scene change */
        wubu_la_destroy(la);
        free(f1);free(f2);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
