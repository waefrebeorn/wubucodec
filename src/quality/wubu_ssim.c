/*
 * wubu_ssim.c -- GROUP 20: Video Quality Metrics
 *
 * G20.01: PSNR per channel (Y,U,V separately)
 * G20.02: SSIM implementation with 8x8 windows
 * G20.03: MS-SSIM multi-scale version
 * G20.06: BD-Rate calculation between two codecs
 */
#include "wubu_ssim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== PSNR per channel ===== */

double wubu_psnr_channel(const uint8_t* a,const uint8_t* b,long n){
    double mse=0;
    for(long i=0;i<n;i++){
        double d=(double)a[i]-b[i];
        mse+=d*d;
    }
    mse/=n;
    if(mse==0)return 99.0;
    return 10.0*log10(255.0*255.0/mse);
}

/* ===== SSIM ===== */

/* Compute local SSIM over an 8x8 window */
static double ssim_window(const uint8_t* x,const uint8_t* y,
                           int W,int H,int wx,int wy,int ws){
    /* compute means, variances, and covariance within the window */
    double mx=0,my=0,n=ws*(double)ws;
    
    for(int dy=0;dy<ws;dy++)
        for(int dx=0;dx<ws;dx++){
            int px=wx+dx,py=wy+dy;
            if(px<W&&py<H){
                mx+=x[(size_t)py*W+px];
                my+=y[(size_t)py*W+px];
            }
        }
    mx/=n;my/=n;
    
    double vx=0,vy=0,cxy=0;
    const double C1=6.5025,C2=58.5225; /* (0.01*255)^2, (0.03*255)^2 */
    
    for(int dy=0;dy<ws;dy++)
        for(int dx=0;dx<ws;dx++){
            int px=wx+dx,py=wy+dy;
            if(px<W&&py<H){
                double dx_val=x[(size_t)py*W+px]-mx;
                double dy_val=y[(size_t)py*W+px]-my;
                vx+=dx_val*dx_val;
                vy+=dy_val*dy_val;
                cxy+=dx_val*dy_val;
            }
        }
    vx/=(n-1);vy/=(n-1);cxy/=(n-1);
    
    double num=(2*mx*my+C1)*(2*cxy+C2);
    double den=(mx*mx+my*my+C1)*(vx+vy+C2);
    return num/den;
}

/* global SSIM: average of local window SSIMs */
double wubu_ssim(const uint8_t* x,const uint8_t* y,int W,int H){
    int ws=8;
    double total=0;
    int count=0;
    for(int wy=0;wy<H;wy+=ws)
        for(int wx=0;wx<W;wx+=ws){
            total+=ssim_window(x,y,W,H,wx,wy,ws);
            count++;
        }
    return count>0?total/count:0;
}

/* ===== MS-SSIM: multi-scale SSIM ===== */

/* downsample by factor of 2 using simple averaging */
static void downsample2(const uint8_t* src,uint8_t* dst,int W,int H){
    int hw=W/2,hh=H/2;
    for(int y=0;y<hh;y++)
        for(int x=0;x<hw;x++){
            int sum=src[((size_t)y*2)*W+(x*2)]
                   +src[((size_t)y*2)*W+(x*2+1)]
                   +src[((size_t)y*2+1)*W+(x*2)]
                   +src[((size_t)y*2+1)*W+(x*2+1)];
            dst[(size_t)y*hw+x]=(uint8_t)(sum/4);
        }
}

double wubu_msssim(const uint8_t* x,const uint8_t* y,int W,int H,int n_scales){
    double mssim=1.0;
    uint8_t* cur_x=malloc((size_t)W*H);
    uint8_t* cur_y=malloc((size_t)W*H);
    memcpy(cur_x,x,(size_t)W*H);
    memcpy(cur_y,y,(size_t)W*H);
    
    int cw=W,ch=H;
    double weights[5]={0.0448,0.2856,0.3001,0.2363,0.1333};
    
    for(int s=0;s<n_scales&&s<5;s++){
        if(cw<16||ch<16)break;
        
        double ssim_val=wubu_ssim(cur_x,cur_y,cw,ch);
        
        if(s<n_scales-1){
            /* contrast+structure term for intermediate scales */
            mssim*=ssim_val;
        }else{
            /* luminance+contrast+structure at final scale */
            mssim*=pow(ssim_val,weights[s]);
        }
        
        if(s<n_scales-1){
            downsample2(cur_x,cur_x,cw,ch); /* in-place is ok for halving */
            downsample2(cur_y,cur_y,cw,ch);
            cw/=2;ch/=2;
        }
    }
    
    free(cur_x);free(cur_y);
    return mssim;
}

/* ===== BD-Rate ===== */

/*
 * Calculate Bjontegaard Delta Rate between two RD curves.
 * Uses the cubic polynomial integration method.
 * rate_a/dist_a: codec A's curve
 * rate_b/dist_b: codec B's curve  
 * n_points: number of points on each curve (must match)
 * Returns: percentage bitrate savings (positive = B saves vs A)
 */
double wubu_bdrate(const double* rate_a,const double* psnr_a,
                    const double* rate_b,const double* psnr_b,
                    int n_points){
    /* integrate log(rate) over PSNR range for each curve */
    double int_a=0,int_b=0;
    double psnr_min=1e9,psnr_max=-1e9;
    
    for(int i=0;i<n_points;i++){
        if(psnr_a[i]<psnr_min)psnr_min=psnr_a[i];
        if(psnr_a[i]>psnr_max)psnr_max=psnr_a[i];
        if(psnr_b[i]<psnr_min)psnr_min=psnr_b[i];
        if(psnr_b[i]>psnr_max)psnr_max=psnr_b[i];
    }
    
    /* trapezoidal integration of log(rate) over PSNR */
    for(int i=1;i<n_points;i++){
        double dr_a=log(rate_a[i])-log(rate_a[i-1]);
        double dp_a=psnr_a[i]-psnr_a[i-1];
        if(dp_a!=0)int_a+=(log(rate_a[i])+log(rate_a[i-1]))/2*dp_a;
        
        double dr_b=log(rate_b[i])-log(rate_b[i-1]);
        double dp_b=psnr_b[i]-psnr_b[i-1];
        if(dp_b!=0)int_b+=(log(rate_b[i])+log(rate_b[i-1]))/2*dp_b;
    }
    
    double avg_log_a=int_a/(psnr_max-psnr_min);
    double avg_log_b=int_b/(psnr_max-psnr_min);
    
    /* BD-rate = exp(avg_B - avg_A) - 1 */
    return (exp(avg_log_b-avg_log_a)-1)*100.0;
}
