#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_part2.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Group 1 Remaining Tests ===\n\n");

    printf("  g1_amp_partitions...");
    {
        wubu_partition_t parts[10];
        int n=wubu_amp_partitions(16,parts);
        printf("[%d AMP partitions] ",n);
        CHECK(n==5); /* 1 standard + 4 asymmetric */
        /* verify the asymmetric partition sizes */
        CHECK(parts[1].h==4);   /* top quarter */
        CHECK(parts[2].h==12);  /* lower 3/4 */
        CHECK(parts[3].w==4);   /* left quarter */
        CHECK(parts[4].w==12);  /* right 3/4 */
    }
    printf("PASS\n");passed++;

    printf("  g2_mv_rounding...");
    {
        int dx=5,dy=-3;
        wubu_mv_round_halfpel(&dx,&dy);
        CHECK(dx==6&&dy==-2); /* rounded to nearest even */
        
        dx=7;dy=9;
        wubu_mv_round_integer(&dx,&dy);
        CHECK(dx==8&&dy==8); /* rounded to nearest multiple of 4 */
    }
    printf("PASS\n");passed++;

    printf("  g3_rd_cost_prefers_small_mvd...");
    {
        /* two candidates with same SAD but different MVD costs */
        double cost_large_mvd=wubu_rd_cost(100,20,-15,0.5);
        double cost_small_mvd=wubu_rd_cost(105,2,-1,0.5);
        printf("[large=%.1f small=%.1f] ",cost_large_mvd,cost_small_mvd);
        /* small SAD increase is worth it if MVD drops significantly */
        CHECK(cost_small_mvd<cost_large_mvd);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
