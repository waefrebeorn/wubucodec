#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_bframe2.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Hierarchical B-Frame Tests ===\n\n");

    printf("  g1_temporal_layers...");
    {
        /* GOP-8: frame 4 is layer 1 (between I(0) and P(8)) */
        int l4=wubu_gop_temporal_layer(4,8);
        int l2=wubu_gop_temporal_layer(2,8);
        
        printf("[f4=layer%d f2=layer%d] ",l4,l2);
        CHECK(l4<=l2); /* frame 4 is at same or shallower layer than 2 */
        CHECK(l4>=0);
    }
    printf("PASS\n");passed++;

    printf("  g2_frame_types...");
    {
        CHECK(wubu_gop_frame_type(0,8)==WUBU_FT_I);
        CHECK(wubu_gop_frame_type(8,8)==WUBU_FT_P);
        wubu_frame_type2_t ft=wubu_gop_frame_type(4,8);
        CHECK(ft==WUBU_FT_B); /* middle frames are B */
    }
    printf("PASS\n");passed++;

    printf("  g3_bipred_average...");
    {
        uint8_t p0[100],p1[100],out[100];
        memset(p0,100,100);memset(p1,200,100);
        wubu_bp_average(p0,p1,out,100);
        CHECK(out[50]==150); /* average of 100 and 200 = 150 */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
