#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_ssim.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Quality Metrics Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_ssim_identical...");
    {
        uint8_t* img=malloc((size_t)W*H);
        for(long i=0;i<(long)W*H;i++)img[i]=(uint8_t)(i%256);
        double s=wubu_ssim(img,img,W,H);
        printf("[SSIM=%.4f] ",s);
        CHECK(s>0.99);
        free(img);
    }
    printf("PASS\n");passed++;

    printf("  g2_ssim_degrades_with_noise...");
    {
        uint8_t* clean=malloc((size_t)W*H);
        uint8_t* noisy=malloc((size_t)W*H);
        for(long i=0;i<(long)W*H;i++)clean[i]=(uint8_t)(i%256);
        memcpy(noisy,clean,(size_t)W*H);
        /* add noise */
        srand(42);
        for(long i=0;i<(long)W*H;i+=7)
            noisy[i]=(uint8_t)((noisy[i]+rand()%20-10)&0xFF);
        
        double s=wubu_ssim(clean,noisy,W,H);
        printf("[SSIM=%.4f] ",s);
        CHECK(s<1.0&&s>0.5); /* degraded but not destroyed */
        free(clean);free(noisy);
    }
    printf("PASS\n");passed++;

    printf("  g3_bdrate_positive_means_savings...");
    {
        /* codec A uses more bits at same quality */
        double rates_a[]={1000,2000,4000,8000};
        double psnrs_a[]={30,33,36,39};
        double rates_b[]={800,1600,3200,6400};  /* 20% less */
        double psnrs_b[]={30,33,36,39};
        
        double bd=wubu_bdrate(rates_a,psnrs_a,rates_b,psnrs_b,4);
        printf("[BD-rate=%.2f%%] ",bd);
        CHECK(bd<-15&&bd>-25); /* negative = B saves */ /* should show ~20% savings */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
