/*
 * wubu_bench_quality.c -- GAP-C052: PSNR + SSIM quality metrics for
 * the WUBQ codec (the measurement layer the A/B needs)
 *
 * Without quality metrics, compression ratios are meaningless — you can
 * always compress more by destroying more. PSNR and SSIM quantify what
 * was lost so the rate-distortion curve is honest.
 */
#include "wubu_bench_quality.h"
#include <math.h>

/* Peak Signal-to-Noise Ratio in dB for RGB frames */
float wubu_q_psnr(const uint8_t* orig,const uint8_t* recon,long n_pixels){
    double mse=0;
    long n=n_pixels*3;
    for(long i=0;i<n;i++){
        double d=orig[i]-recon[i];
        mse+=d*d;
    }
    mse/=n;
    if(mse<=0)return 99.0f;   /* identical */
    return 10.0f*log10(255.0*255.0/mse);
}

/* SSIM on luminance (approximate: global SSIM, not windowed) */
float wubu_q_ssim(const uint8_t* a,const uint8_t* b,long n){
    /* convert to luminance approximation and compute global SSIM */
    double mu_a=0,mu_b=0;
    for(long i=0;i<n;i++){mu_a+=a[i*3];mu_b+=b[i*3];}
    mu_a/=n;mu_b/=n;

    double var_a=0,var_b=0,cov=0;
    const double C1=6.5025,C2=58.5225;  /* (0.01*255)^2, (0.03*255)^2 */
    for(long i=0;i<n;i++){
        double da=a[i*3]-mu_a,db=b[i*3]-mu_b;
        var_a+=da*da;var_b+=db*db;
        cov+=da*db;
    }
    var_a/=n-1;var_b/=n-1;cov/=(n-1);

    double num=(2*mu_a*mu_b+C1)*(2*cov+C2);
    double den=(mu_a*mu_a+mu_b*mu_b+C1)*(var_a+var_b+C2);
    return (float)(num/den);
}

/* mean absolute per-channel error */
float wubu_q_mae(const uint8_t* orig,const uint8_t* recon,long n_pixels){
    double sum=0;
    long n=n_pixels*3;
    for(long i=0;i<n;i++)
        sum+=fabs((double)orig[i]-recon[i]);
    return (float)(sum/n);
}

/* full report for one frame pair */
void wubu_q_report(const uint8_t* orig,const uint8_t* recon,
                    int W,int H,float* psnr,float* ssim,float* mae){
    long np=(long)W*H;
    *psnr=wubu_q_psnr(orig,recon,np);
    *ssim=wubu_q_ssim(orig,recon,np);
    *mae=wubu_q_mae(orig,recon,np);
}
