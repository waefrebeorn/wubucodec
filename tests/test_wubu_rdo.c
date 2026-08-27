#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_rdo.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== RDO Framework Tests ===\n\n");

    printf("  g1_lambda_increases_with_qp...");
    {
        double l1=wubu_lambda_from_qp(20,WUBU_P_FRAME);
        double l2=wubu_lambda_from_qp(30,WUBU_P_FRAME);
        printf("[QP20=%.4f QP30=%.4f] ",l1,l2);
        CHECK(l2>l1); /* higher QP → more willing to accept distortion */
    }
    printf("PASS\n");passed++;

    printf("  g2_skip_selected_when_identical...");
    {
        /* identical prediction and original → SKIP should win */
        uint8_t* img=malloc(256);
        memset(img,128,256);
        int16_t coeffs[64];memset(coeffs,0,sizeof(coeffs));
        
        wubu_mode_t mode=wubu_best_mode(img,img,img,coeffs,256,64,0.5);
        printf("[mode=%d] ",mode);
        CHECK(mode==WUBU_MODE_SKIP);
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g3_early_termination...");
    {
        /* low SAD + low variance → can terminate early */
        CHECK(wubu_can_early_terminate_skip(5,10,100)==1);
        /* high SAD → cannot skip */
        CHECK(wubu_can_early_terminate_skip(500,10,100)==0);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
