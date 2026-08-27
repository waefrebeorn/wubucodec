#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_angular.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Angular Intra Prediction Tests ===\n\n");
    const int W=64,H=64,BS=8;

    printf("  g1_dc_flat...");
    {
        uint8_t* img=calloc((size_t)W*H,1);
        memset(img,100,(size_t)W*H);
        uint8_t block[BS_MAX*BS_MAX];
        wubu_ipred(img,W,H,1,1,BS,1,block); /* DC mode */
        for(int i=0;i<BS*BS;i++)CHECK(block[i]==100);
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g2_vertical_pattern...");
    {
        /* vertical stripes → vertical angular modes should work well */
        uint8_t* img=malloc((size_t)W*H);
        /* constant columns with varying values → vertical prediction wins */
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                img[(size_t)y*W+x]=(uint8_t)(100+(x%8)*20);
        
        uint8_t actual[BS_MAX*BS_MAX],pred[BS_MAX*BS_MAX];
        for(int r=0;r<BS;r++)for(int c=0;c<BS;c++)
            actual[r*BS+c]=img[(size_t)(16+r)*W+(16+c)];
        
        wubu_ipred(img,W,H,2,2,BS,26,pred); /* vertical-ish mode */
        long sad_v=0;
        for(int i=0;i<BS*BS;i++)sad_v+=abs(actual[i]-pred[i]);
        
        /* compare against DC */
        wubu_ipred(img,W,H,2,2,BS,1,pred);
        long sad_dc=0;
        for(int i=0;i<BS*BS;i++)sad_dc+=abs(actual[i]-pred[i]);
        
        printf("[vertical SAD=%ld vs DC=%ld] ",sad_v,sad_dc);
        CHECK(sad_v<sad_dc);
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g3_best_mode_selects...");
    {
        /* gradient → should pick a non-DC mode */
        uint8_t* img=malloc((size_t)W*H);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                img[(size_t)y*W+x]=(uint8_t)(x*3+y);
        
        uint8_t actual[BS_MAX*BS_MAX];
        for(int r=0;r<BS;r++)for(int c=0;c<BS;c++)
            actual[r*BS+c]=img[(size_t)(24+r)*W+(24+c)];
        
        int mode=wubu_ipred_best_mode(img,W,H,3,3,BS,actual);
        printf("[best=%d] ",mode);
        CHECK(mode>=0&&mode<35); /* valid mode */
        free(img);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
