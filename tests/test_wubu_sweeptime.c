#include <stdio.h>
#include <math.h>
#include "wubu_sweeptime.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Sweep Time Tests ===\n\n");
    printf("  g1_roundtrip...");
    {
        float dur=10.0f;int W=1920;
        for(int x=0;x<W;x+=97){
            float t=wubu_st_x_to_t(x,W,dur);
            int back=wubu_st_t_to_x(t,dur,W);
            CHECK(abs(back-x)<=1);
        }
    }
    printf("PASS\n");passed++;
    printf("  g2_endpoint...");
    {
        CHECK(wubu_st_x_to_t(0,100,5)==0);
        CHECK(fabsf(wubu_st_x_to_t(99,100,5)-5.0f)<0.1f);
    }
    printf("PASS\n");passed++;
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
