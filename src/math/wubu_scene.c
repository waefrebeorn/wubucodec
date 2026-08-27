/*
 * wubu_scene.c -- GROUP 16: Scene analysis & adaptive encoding
 *
 * G16.01: Scene change detection (histogram difference)
 * G16.02: Content complexity (spatial + temporal activity)
 * G16.03: Adaptive QP from complexity
 * G16.07: CRF rate control mode
 */
#include "wubu_scene.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G16.01: Scene Change Detection ===== */

/* Compute 32-bin luminance histogram */
void wubu_histogram(const uint8_t* img,long n,uint32_t* hist){
    memset(hist,0,sizeof(uint32_t)*32);
    for(long i=0;i<n;i++)
        hist[img[i]>>3]++;
}

/* Normalized histogram intersection distance: 0=identical, 1=different */
double wubu_hist_distance(const uint32_t* h1,const uint32_t* h2,long total){
    double diff=0;
    for(int i=0;i<32;i++){
        double a=(double)h1[i]/total;
        double b=(double)h2[i]/total;
        diff+=fabs(a-b);
    }
    return diff/2; /* normalize to [0,1] */
}

/* Detect scene change between two frames */
int wubu_scene_change(const uint8_t* prev,const uint8_t* curr,
                       int W,int H,double threshold){
    long n=(long)W*H;
    uint32_t hist_prev[32],hist_curr[32];
    
    wubu_histogram(prev,n,hist_prev);
    wubu_histogram(curr,n,hist_curr);
    
    double dist=wubu_hist_distance(hist_prev,hist_curr,n);
    return dist>threshold;
}

/* ===== G16.02: Content Complexity ===== */

/* Spatial activity: average edge strength using Sobel-like gradient */
double wubu_spatial_complexity(const uint8_t* img,int W,int H){
    if(W<3||H<3)return 0;
    long sum=0,count=0;
    for(int y=1;y<H-1;y++)
        for(int x=1;x<W-1;x++){
            int gx=img[(size_t)y*W+x+1]-img[(size_t)y*W+x-1];
            int gy=img[(size_t)(y+1)*W+x]-img[(size_t)(y-1)*W+x];
            sum+=(long)abs(gx)+abs(gy);
            count++;
        }
    return count>0?(double)sum/count:0;
}

/* Temporal activity: mean absolute pixel difference between frames */
double wubu_temporal_complexity(const uint8_t* prev,const uint8_t* curr,
                                  long n){
    double sum=0;
    for(long i=0;i<n;i++)
        sum+=abs(prev[i]-curr[i]);
    return sum/n;
}

/* ===== G16.03: Adaptive QP from complexity ===== */

/*
 * Adjust QP based on content:
 * - High spatial detail → lower QP (more bits to preserve detail)
 * - Low complexity → higher QP (bits won't be missed)
 * Returns QP offset from base.
 */
int wubu_adaptive_qp(double spatial_act,double temporal_act,
                      int base_qp,int max_offset){
    int offset=0;
    
    /* high spatial detail → reduce QP by up to max_offset/2 */
    if(spatial_act>20)offset-=max_offset/2;
    else if(spatial_act>10)offset-=max_offset/4;
    
    /* high temporal activity → slightly increase QP (motion masks artifacts) */
    if(temporal_act>15)offset+=max_offset/4;
    else if(temporal_act<5)offset-=max_offset/4;
    
    return offset;
}

/* ===== G16.07: Simple CRF rate control ===== */

void wubu_crf_init(WubuCrfState* crf,double crf_value){
    crf->quality_factor=crf_value;
    crf->last_qp=23; /* default starting QP */
}

/* update QP based on actual vs expected bits */
int wubu_crf_update(WubuCrfState* crf,long actual_bits,
                     long target_bits,int current_qp){
    double ratio=(double)actual_bits/target_bits;
    
    /* adjust QP to hit the rate target */
    if(ratio>1.2)crf->last_qp=current_qp+2;   /* too many bits → raise QP */
    else if(ratio>1.05)crf->last_qp=current_qp+1;
    else if(ratio<0.8)crf->last_qp=current_qp-2; /* too few bits → lower QP */
    else if(ratio<0.95)crf->last_qp=current_qp-1;
    else crf->last_qp=current_qp; /* on target */
    
    /* clamp QP to valid range */
    if(crf->last_qp<10)crf->last_qp=10;
    if(crf->last_qp>51)crf->last_qp=51;
    
    return crf->last_qp;
}
