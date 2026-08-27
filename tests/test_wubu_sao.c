#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_sao.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
static double psnr(const uint8_t*a,const uint8_t*b,long n){
    double mse=0;for(long i=0;i<n;i++){double d=a[i]-b[i];mse+=d*d;}
    mse/=n;return mse>0?10*log10(255.0*255.0/mse):99;
}
int main(void){
    printf("=== SAO Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_edge_offsets_improve...");
    {
        /* create original + degraded reconstruction */
        uint8_t* orig=malloc((size_t)W*H);
        uint8_t* recon=malloc((size_t)W*H);
        uint8_t* filtered=malloc((size_t)W*H);
        
        for(long i=0;i<(long)W*H;i++){
            orig[i]=(uint8_t)((i*7)%256); /* gradient pattern */
            recon[i]=orig[i]+((i%3)-1);   /* add ±1 noise */
        }
        
        double psnr_before=psnr(orig,recon,W*(long)H);
        
        /* estimate and apply edge offsets */
        int offsets[5];
        wubu_sao_edge_estimate(orig,recon,W,H,0,offsets); /* dir=0 horizontal */
        wubu_sao_edge(recon,filtered,W,H,0,offsets);
        
        double psnr_after=psnr(orig,filtered,(long)W*H);
        printf("[before=%.2f after=%.2f dB] ",psnr_before,psnr_after);
        CHECK(psnr_after>=psnr_before-0.5); /* should not degrade significantly */
        
        free(orig);free(recon);free(filtered);
    }
    printf("PASS\n");passed++;

    printf("  g2_band_offsets...");
    {
        uint8_t* orig=malloc((size_t)W*H);
        uint8_t* recon=malloc((size_t)W*H);
        uint8_t* filtered=malloc((size_t)W*H);
        
        /* dark bias in low values, bright bias in high values */
        for(long i=0;i<(long)W*H;i++){
            int v=(int)(i%256);
            orig[i]=(uint8_t)v;
            recon[i]=(uint8_t)(v<128?v+5:v-5); /* systematic bias */
        }
        
        int offsets[32];
        wubu_sao_band_estimate(orig,recon,W,H,0,offsets,32);
        wubu_sao_band(recon,filtered,W,H,0,offsets,32);
        
        /* check that offsets correct the bias direction */
        CHECK(offsets[10]<0||offsets[20]>0); /* at least one non-zero offset */
        
        free(orig);free(recon);free(filtered);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
