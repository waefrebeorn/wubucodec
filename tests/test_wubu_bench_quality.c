#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_bench_quality.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Quality Metrics Tests ===\n\n");
    const int W=64,H=48;
    long np=(long)W*H;
    uint8_t* a=malloc(np*3);
    uint8_t* b=malloc(np*3);

    printf("  g1_identical_infinite_psnr...");
    {
        for(long i=0;i<np*3;i++)a[i]=b[i]=(uint8_t)(i%256);
        float p=wubu_q_psnr(a,b,np);
        CHECK(p>90.0f);
    }
    printf("PASS\n");passed++;

    printf("  g2_noise_reduces_psnr...");
    {
        for(long i=0;i<np*3;i++)a[i]=128;
        memcpy(b,a,np*3);
        /* add small noise */
        float psnr_small=wubu_q_psnr(a,b,np);
        for(long i=0;i<np*3;i++)b[i]=128+((i%7)-3);
        float psnr_big=wubu_q_psnr(a,b,np);
        CHECK(psnr_big<psnr_small);
    }
    printf("PASS\n");passed++;

    printf("  g3_ssim_range...");
    {
        /* identical → SSIM ≈ 1 */
        for(long i=0;i<np*3;i++){a[i]=100+(i%50);b[i]=a[i];}
        float s=wubu_q_ssim(a,b,np);
        CHECK(s>0.99f);
        /* inverted → SSIM < 0 */
        for(long i=0;i<np*3;i++)b[i]=255-a[i];
        s=wubu_q_ssim(a,b,np);
        CHECK(s<0.5f);
    }
    printf("PASS\n");passed++;

    free(a);free(b);
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
