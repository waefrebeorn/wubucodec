/*
 * wubu_neural_codec.c -- GROUP 25: Neural codec building blocks
 *
 * G25.03: Entropy model (learned probability estimation)
 * G25.04: Latent representation with scale hyperprior
 * G25.05: Arithmetic coding with neural probability estimates
 * G25.06: Rate-distortion loss function
 *
 * Research source: Ballé et al. 2018 "Variational image compression
 * with a scale hyperprior" — the foundational neural codec paper.
 *
 * This is NOT a full neural codec (needs GPU + training). These are the
 * C11 inference-side building blocks that a trained model would use.
 */
#define M_PI 3.14159265358979f
#include "wubu_ncodec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G25.03: Learned Entropy Model ===== */

/*
 * Scale hyperprior entropy model:
 * Each latent has a predicted scale σ from the hyperprior network.
 * Probability of each quantized latent value uses a logistic distribution
 * centered at 0 with learned scale σ.
 *
 * P(x) = ∫ Normal(0,σ²) · Uniform(x-0.5,x+0.5) dx
 *      ≈ Φ((x+0.5)/σ) - Φ((x-0.5)/σ)
 * where Φ is the logistic CDF.
 */

double wubu_nc_logistic_cdf(double x,double mean,double scale){
    return 1.0/(1.0+exp(-(x-mean)/scale));
}

/* probability mass for quantized value x given scale σ */
double wubu_nc_prob_mass(int x_val,double scale){
    double lower=wubu_nc_logistic_cdf((double)x_val-0.5,0,scale);
    double upper=wubu_nc_logistic_cdf((double)x_val+0.5,0,scale);
    double p=upper-lower;
    if(p<1e-10)p=1e-10;
    return p;
}

/* bits needed to code one latent = -log2(P(x)) */
double wubu_nc_bits_per_symbol(int x_val,double scale){
    return -log2(wubu_nc_prob_mass(x_val,scale));
}

/* total rate estimate for all latents */
double wubu_nc_rate_estimate(const int16_t* latents,const double* scales,
                               long n_latents){
    double total_bits=0;
    for(long i=0;i<n_latents;i++)
        total_bits+=wubu_nc_bits_per_symbol(latents[i],scales[i]);
    return total_bits;
}

/* ===== G25.06: Rate-Distortion Loss ===== */

/*
 * L = λ·D + R where:
 *   D = MSE between original and reconstruction  
 *   R = rate from entropy model above
 *   λ controls the rate-distortion tradeoff
 */
double wubu_nc_rd_loss(const float* original,const float* reconstructed,
                         const int16_t* latents,const double* scales,
                         long n_pixels,long n_latents,
                         double lambda){
    /* distortion: MSE */
    double mse=0;
    for(long i=0;i<n_pixels;i++){
        double d=original[i]-reconstructed[i];
        mse+=d*d;
    }
    mse/=n_pixels;
    
    /* rate from entropy model */
    double rate=wubu_nc_rate_estimate(latents,scales,n_latents);
    
    /* normalized per pixel */
    rate/=n_pixels;
    
    return lambda*mse+rate;
}

/* ===== G25.04: Simplified hyperprior ===== */

/*
 * The hyperprior provides side information (scale parameters) that helps
 * the main entropy model predict better probabilities.
 * In practice this is a small neural net; here we compute it as a simple
 * spatial average of neighboring scales.
 */
void wubu_nc_hyperprior_scale(const int16_t* latents,long W,long H,
                                double base_scale,double* out_scales){
    /* local variance → local scale */
    for(long y=1;y<H-1;y++)
        for(long x=1;x<W-1;x++){
            long idx=y*W+x;
            
            /* measure local activity from neighbors */
            double var=0;
            var+=(double)(latents[idx]-latents[idx-1])*(latents[idx]-latents[idx-1]);
            var+=(double)(latents[idx]-latents[idx+1])*(latents[idx]-latents[idx+1]);
            var+=(double)(latents[idx]-latents[idx-W])*(latents[idx]-latents[idx-W]);
            var+=(double)(latents[idx]-latents[idx+W])*(latents[idx]-latents[idx+W]);
            var/=4;
            
            /* scale proportional to sqrt(local variance), clamped */
            double s=sqrt(var)*0.5;
            if(s<base_scale)s=base_scale;
            if(s>base_scale*4)s=base_scale*4;
            
            out_scales[idx]=s;
        }
}

/* ===== G25.05: Range coder with adaptive probability ===== */

/*
 * Simple binary arithmetic coder driven by the entropy model's
 * estimated probabilities. This replaces CABAC's fixed state machine
 * with learned, continuous probabilities.
 */

void wubu_nc_arith_init(NCArithEnc* e,uint8_t* buf,size_t cap){
    e->low=0;e->range=0xFFFFFFFF;
    e->buf=buf;e->cap=cap;e->bitpos=0;
}

static void nc_shift_low(NCArithEnc* e){
    while(e->range<=0xFFFFFF){
        int bit=(e->low>>32)&1;
        size_t byte=e->bitpos/8;
        int shift=7-(int)(e->bitpos%8);
        if(byte<e->cap){
            if(bit)e->buf[byte]|=(1<<shift);
        }
        e->bitpos++;
        e->low<<=1;e->range<<=1;
    }
}

void wubu_nc_arith_encode(NCArithEnc* e,int bit,double prob_one){
    /* subdivide range by prob_one */
    uint32_t range_one=(uint32_t)((uint64_t)e->range*prob_one);
    uint32_t range_zero=e->range-range_one;
    if(range_one==0)range_one=1;
    if(range_zero==0)range_zero=1;
    
    if(bit){
        e->low+=range_zero;
        e->range=range_one;
    }else{
        e->range=range_zero;
    }
    nc_shift_low(e);
}

long wubu_nc_arith_finish(NCArithEnc* e){
    /* flush remaining bits */
    for(int i=0;i<4;i++){
        int bit=(e->low>>32)&1;
        size_t byte=e->bitpos/8;
        int shift=7-(int)(e->bitpos%8);
        if(byte<e->cap&&bit)e->buf[byte]|=(1<<shift);
        e->bitpos++;e->low<<=1;
    }
    return (long)((e->bitpos+7)/8);
}
