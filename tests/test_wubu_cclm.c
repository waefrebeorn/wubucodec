#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_cclm.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== CCLM + JCCR Tests ===\n\n");

    printf("  g1_cclm_fit_linear...");
    {
        /* chroma = 2×luma + 10 → fit should recover α≈2, β≈10 */
        uint8_t luma[16]={10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85};
        uint8_t chroma[16];
        for(int i=0;i<16;i++)chroma[i]=(uint8_t)(2*luma[i]+10);
        
        double alpha,beta;
        int ret=wubu_cclm_fit(luma,chroma,16,&alpha,&beta);
        CHECK(ret==0);
        printf("[α=%.2f β=%.2f] ",alpha,beta);
        CHECK(fabs(alpha-2.0)<0.1);
        CHECK(fabs(beta-10.0)<5.0);
    }
    printf("PASS\n");passed++;

    printf("  g2_jccr_same_sign...");
    {
        /* same-sign, similar magnitude residuals → JCCR applicable */
        int16_t cb[]={10,20,-5,30};
        int16_t cr[]={12,-18,6,25};
        CHECK(wubu_jccr_check(cb,cr,4,2.0)==0); /* signs don't match at [1] */
        
        int16_t cb2[]={10,20,-5,30};
        int16_t cr2[]={11,19,-4,28};
        CHECK(wubu_jccr_check(cb2,cr2,4,2.0)==1); /* all match */
    }
    printf("PASS\n");passed++;

    printf("  g3_jccr_split_roundtrip...");
    {
        int16_t joint[4]={15,-20,8,0};
        int16_t cb[4],cr[4];
        
        wubu_jccr_split(joint,cb,cr,4,0); /* mode 0: cb=cr */
        for(int i=0;i<4;i++)CHECK(cb[i]==joint[i]&&cr[i]==joint[i]);
        
        wubu_jccr_split(joint,cb,cr,4,1); /* mode 1: cr=-joint */
        for(int i=0;i<4;i++)CHECK(cb[i]==joint[i]&&cr[i]==-joint[i]);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
