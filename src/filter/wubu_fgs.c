/*
 * wubu_fgs.c -- GROUP 13: Film Grain Synthesis
 *
 * G13.01: Grain parameter estimation (AR coefficients + scaling)
 * G13.02: Noise template generation (64x64 autoregressive)
 * G13.03: Scaling function for luma-dependent grain strength
 *
 * Research source: AV1 FGS (Netflix/DCC 2018), VVC FGC-SEI
 *
 * The idea: denoise before encoding, transmit grain parameters as
 * metadata, synthesize grain at decode. Grain masks compression
 * artifacts and preserves artistic look at lower bitrates.
 */
#define M_PI 3.14159265358979f
#include "wubu_fgs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G13.01: Parameter estimation ===== */

/*
 * Estimate AR(3) coefficients from the grain residual
 * (source minus denoised). Uses Yule-Walker on the residual.
 */
void wubu_fgs_estimate_ar(const int16_t* residual,long n_samples,
                            double* ar_coeffs){
    /* compute autocorrelation at lags 0-3 */
    double ac[4]={0,0,0,0};
    for(long i=3;i<n_samples;i++){
        ac[0]+=(double)residual[i]*residual[i];
        ac[1]+=(double)residual[i]*residual[i-1];
        ac[2]+=(double)residual[i]*residual[i-2];
        ac[3]+=(double)residual[i]*residual[i-3];
    }
    for(int k=0;k<4;k++)ac[k]/=(n_samples-3);
    
    /* solve 3x3 Yule-Walker equations via simple matrix inversion */
    /* [ac0 ac1 ac2] [a1]   [ac1]
       [ac1 ac0 ac1] [a2] = [ac2]
       [ac2 ac1 ac0] [a3]   [ac3] */
    
    if(ac[0]<1e-10){ar_coeffs[0]=ar_coeffs[1]=ar_coeffs[2]=0;return;}
    
    /* Cramer's rule */
    double det=ac[0]*(ac[0]*ac[0]-ac[1]*ac[1])
              -ac[1]*(ac[1]*ac[0]-ac[1]*ac[2])
              +ac[2]*(ac[1]*ac[1]-ac[0]*ac[2]);
    if(fabs(det)<1e-10){ar_coeffs[0]=ar_coeffs[1]=ar_coeffs[2]=0;return;}
    
    ar_coeffs[0]=((ac[1])*(ac[0]*ac[0]-ac[1]*ac[1])
                -(ac[0]*ac[1]-ac[2]*ac[1])*(ac[1]*ac[0]-ac[1]*ac[2])
                +(ac[2]*ac[1]-ac[1]*ac[1])*(ac[1]*ac[1]-ac[0]*ac[2]))/det;
    ar_coeffs[1]=((ac[0])*(ac[2]*ac[0]-ac[1]*ac[2])
                -(ac[1]*ac[1]-ac[1]*ac[2])*(ac[1]*ac[0]-ac[1]*ac[2])
                +(ac[2]*ac[2]-ac[1]*ac[1])*(ac[1]*ac[1]-ac[0]*ac[2]))/det;
    ar_coeffs[2]=((ac[0])*(ac[0]*ac[1]-ac[1]*ac[0])
                -(ac[1])*(ac[1]*ac[0]-ac[1]*ac[2])
                +(ac[2])*(ac[1]*ac[1]-ac[0]*ac[2]))/det;
    
    /* normalize to prevent instability */
    double sum=fabs(ar_coeffs[0])+fabs(ar_coeffs[1])+fabs(ar_coeffs[2]);
    if(sum>0.95){
        double scale=0.95/sum;
        ar_coeffs[0]*=scale;ar_coeffs[1]*=scale;ar_coeffs[2]*=scale;
    }
}

/* estimate scaling function: grain intensity vs luma value */
void wubu_fgs_estimate_scaling(const uint8_t* source,const uint8_t* denoised,
                                 long n_pixels,int* scaling_lut){
    /* bin by luma value (32 bins of 8) */
    long count[32]={0};
    double var_sum[32]={0};
    
    for(long i=0;i<n_pixels;i++){
        int luma=denoised[i]>>3;
        if(luma>=32)luma=31;
        int noise=source[i]-denoised[i];
        var_sum[luma]+=(double)noise*noise;
        count[luma]++;
    }
    
    /* scaling = sqrt(variance), clamped to [-255,255] range */
    for(int b=0;b<32;b++){
        if(count[b]>10)
            scaling_lut[b]=(int)(sqrt(var_sum[b]/count[b])*2.0);
        else
            scaling_lut[b]=8; /* default moderate grain */
        
        if(scaling_lut[b]<1)scaling_lut[b]=1;
        if(scaling_lut[b]>128)scaling_lut[b]=128;
    }
}

/* ===== G13.02: Noise template generation ===== */

/*
 * Generate a 64×64 noise template using the AR model.
 * Each pixel is predicted from its already-generated neighbors
 * plus white Gaussian noise.
 */
void wubu_fgs_gen_template(const double* ar_coeffs,uint8_t* template_64x64,
                             unsigned int seed){
    srand(seed);
    
    /* generate with signed values first, then shift to [0,255] */
    float noise[64*64];
    memset(noise,0,sizeof(noise));
    
    float wgn_std=20.0f;
    for(long i=0;i<(long)64*64;i++){
        /* white Gaussian noise component */
        float wgn=wgn_std*((float)rand()/RAND_MAX-0.5f)*2.0f;
        
        /* AR prediction from left, top, top-left, top-right neighbors */
        float pred=0;
        if(i%64>0)pred+=ar_coeffs[0]*noise[i-1];
        if(i>=64)pred+=ar_coeffs[1]*noise[i-64];
        if(i%64>0&&i>=65)pred+=ar_coeffs[2]*noise[i-65];
        if(i%64<63&&i>=63)pred+=ar_coeffs[3-1+1>2?2:2]*noise[i-63];/*top-right*/
        
        noise[i]=wgn+pred;
        
        /* clamp to prevent runaway */
        if(noise[i]>100)noise[i]=100;
        if(noise[i]<-100)noise[i]=-100;
    }
    
    /* find min/max to normalize to [0,255] */
    float mn=noise[0],mx=noise[0];
    for(long i=1;i<(long)64*64;i++){
        if(noise[i]<mn)mn=noise[i];
        if(noise[i]>mx)mx=noise[i];
    }
    
    float range=mx-mn;
    if(range<1e-6f)range=1.0f;
    
    for(long i=0;i<(long)64*64;i++)
        template_64x64[i]=(uint8_t)((noise[i]-mn)/range*255);
}

/* ===== Apply grain to decoded video ===== */

void wubu_fgs_apply(uint8_t* decoded,const uint8_t* template_64x64,
                      const int* scaling_lut,long W,long H,unsigned int frame_seed){
    unsigned int rng=frame_seed;
    
    for(long y=0;y<H;y++)
        for(long x=0;x<W;x++){
            /* pick random offset into template */
            rng=rng*1103515245+12345;
            int tx=(rng>>16)%56; /* 64-8 so we can grab an 8x8 patch */
            rng=rng*1103515245+12345;
            int ty=(rng>>16)%56;
            
            int ty_off=((y>>3)+ty)%56;
            int tx_off=((x>>3)+tx)%56;
            
            uint8_t grain=template_64x64[(size_t)ty_off*64+tx_off];
            int luma_bin=decoded[y*W+x]>>3;
            
            /* scale grain intensity based on luma */
            int scaled_grain=((int)grain-128)*scaling_lut[luma_bin]/64;
            
            int out=decoded[y*W+x]+scaled_grain;
            decoded[y*W+x]=(uint8_t)(out<0?0:(out>255?255:out));
        }
}
