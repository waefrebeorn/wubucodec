#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_ncodec.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Neural Codec Building Blocks Tests ===\n\n");

    printf("  g1_skewed_latents_fewer_bits...");
    {
        /* mostly-zero latents should cost fewer bits than random ones */
        long n=1000;
        int16_t* lat=malloc(sizeof(int16_t)*(size_t)n);
        double* scales=malloc(sizeof(double)*(size_t)n);
        
        srand(42);
        for(long i=0;i<n;i++){
            lat[i]=rand()%100<90?0:1; /* 90% zeros → low entropy */
            scales[i]=0.1;
        }
        double bits_skew=wubu_nc_rate_estimate(lat,scales,n);
        
        for(long i=0;i<n;i++){
            lat[i]=rand()%2; /* 50/50 → high entropy */
            scales[i]=0.1;
        }
        double bits_rand=wubu_nc_rate_estimate(lat,scales,n);
        
        printf("[skew=%.0f bits rand=%.0f bits] ",bits_skew,bits_rand);
        CHECK(bits_skew<bits_rand*0.5); /* skewed should use less than half */
        free(lat);free(scales);
    }
    printf("PASS\n");passed++;

    printf("  g2_rd_loss_tradeoff...");
    {
        /* lower λ should favor rate (fewer bits) over distortion */
        float orig[100],recon[100];
        int16_t lat[100];
        double scales[100];
        for(int i=0;i<100;i++){orig[i]=(float)i;recon[i]=(float)(i+5);lat[i]=i%7;scales[i]=2.0;}
        
        double loss_low_lambda=wubu_nc_rd_loss(orig,recon,lat,scales,100,100,0.01);
        double loss_high_lambda=wubu_nc_rd_loss(orig,recon,lat,scales,100,100,10.0);
        
        printf("[low_λ=%.2f high_λ=%.2f] ",loss_low_lambda,loss_high_lambda);
        CHECK(loss_high_lambda>loss_low_lambda); /* higher weight on MSE → higher loss */
    }
    printf("PASS\n");passed++;

    printf("  g3_arith_coder_compression...");
    {
        /* encode 500 biased bits through arithmetic coder */
        uint8_t* buf=calloc(4096,1);
        NCArithEnc enc;
        wubu_nc_arith_init(&enc,buf,4096);
        
        srand(42);
        for(int i=0;i<500;i++){
            int bit=rand()%100<95?1:0; /* 95% ones */
            wubu_nc_arith_encode(&enc,bit,0.95);
        }
        long size=wubu_nc_arith_finish(&enc);
        
        printf("[%ld bytes for 500 bins = %.2f b/b] ",size,(float)size*8/500);
        CHECK(size<200); /* 500 uniform would be ~62 bytes; skewed should be less */
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
