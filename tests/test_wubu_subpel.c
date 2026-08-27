#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_subpel.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Sub-pel Motion Estimation Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_halfpel_preserves_originals...");
    {
        uint8_t* src=malloc((size_t)W*H);
        for(long i=0;i<(long)W*H;i++)src[i]=(uint8_t)((i*7)%256);
        uint8_t* hp=malloc((size_t)W*2*H*2);
        wubu_sp_halfpel(src,hp,W,H);
        
        /* original pixels should appear at even-even positions unchanged */
        int match=1;
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                if(hp[(size_t)(y*2)*(W*2)+(x*2)]!=src[(size_t)y*W+x])
                    {match=0;break;}
        CHECK(match);
        free(src);free(hp);
    }
    printf("PASS\n");passed++;

    printf("  g2_halfpel_smooth_gradient...");
    {
        /* smooth gradient → half-pel should be between neighbors */
        uint8_t* src=malloc((size_t)W*H);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                src[(size_t)y*W+x]=(uint8_t)(x*3+y);
        
        uint8_t* hp=malloc((size_t)W*2*H*2);
        wubu_sp_halfpel(src,hp,W,H);

        /* check horizontal half-pel at (x+0.5,y): should be ~average of x and x+1 */
        int ok=1;
        for(int y=10;y<H-2;y++)
            for(int x=5;x<W-5;x++){
                float expected=(src[(size_t)y*W+x]+src[(size_t)y*W+x+1])/2.0f;
                float actual=hp[(size_t)(y*2)*(W*2)+(x*2+1)];
                if(fabsf(actual-expected)>15.0f){ok=0;break;}
            }
        CHECK(ok);
        free(src);free(hp);
    }
    printf("PASS\n");passed++;

    printf("  g3_subpel_me_finds_translation...");
    {
        /* create a frame shifted by exactly 2.5 pixels right */
        uint8_t* ref=malloc((size_t)W*H);
        uint8_t* curr=malloc((size_t)W*H);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                ref[(size_t)y*W+x]=(uint8_t)((x*x/7+y*y/5+x*y/3)%256);
        /* curr = shift ref right by 2 pixels */
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                int sx=x-2;
                if(sx<0)sx=0;if(sx>=W)sx=W-1;
                curr[(size_t)y*W+x]=ref[(size_t)y*W+sx];
            }
        
        /* build half-pel reference */
        uint8_t* hp=malloc((size_t)W*2*H*2);
        wubu_sp_halfpel(ref,hp,W,H);

        int mv_dx,mv_dy;
        long sad=wubu_sp_me(curr,hp,W*2,H*2,16,16,8,8,&mv_dx,&mv_dy);
        
        /* expect dx≈2 in integer pel = 8 quarter-pels */
        printf("[MV=(%d,%d) qpel, SAD=%ld] ",mv_dx,mv_dy,sad);
        CHECK(abs(mv_dx/4-(-2))<=1); /* shifted left means MV points to source */
        free(ref);free(curr);free(hp);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
