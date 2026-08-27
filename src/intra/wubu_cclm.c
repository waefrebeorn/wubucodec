/*
 * wubu_cclm.c -- GROUP 10: Cross-Component Linear Model (CCLM)
 * + Joint Chroma Residual (JCCR)
 *
 * G10.08: CCLM — predicts chroma from luma using a fitted linear model.
 *   chroma = α × luma + β
 * α and β are derived from neighboring reconstructed luma/chroma samples.
 *
 * G10.09: JCCR — when both Cb and Cr residuals have the same sign and
 * similar magnitude, code a single joint residual instead of two separate.
 * Saves ~30% of chroma residual bits in flat-color regions (cartoons!).
 */
#define M_PI 3.14159265358979f
#include "wubu_cclm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G10.08: Cross-Component Linear Model ===== */

/*
 * Fit a linear model: chroma ≈ α × luma + β
 * Using least squares over available neighboring samples.
 * Returns 0 on success, fills alpha/beta.
 */
int wubu_cclm_fit(const uint8_t* luma,const uint8_t* chroma,
                    int n_samples,double* alpha,double* beta){
    if(n_samples<4)return -1;
    
    /* least squares: minimize Σ(chroma - α*luma - β)² */
    double sum_l=0,sum_c=0,sum_ll=0,sum_lc=0;
    
    for(int i=0;i<n_samples;i++){
        double l=luma[i],c=chroma[i];
        sum_l+=l;sum_c+=c;
        sum_ll+=l*l;sum_lc+=l*c;
    }
    
    double denom=n_samples*sum_ll-sum_l*sum_l;
    if(fabs(denom)<1e-10){
        /* luma is constant → use mean as prediction, α=0 */
        *alpha=0;
        *beta=sum_c/n_samples;
        return 0;
    }
    
    *alpha=(n_samples*sum_lc-sum_l*sum_c)/denom;
    *beta=(sum_c-*alpha*sum_l)/n_samples;
    return 0;
}

/* predict chroma block from corresponding luma block */
void wubu_cclm_predict(const uint8_t* luma_block,int bs,
                         double alpha,double beta,uint8_t* chroma_out){
    for(int i=0;i<bs*bs;i++){
        int val=(int)(alpha*luma_block[i]+beta+0.5);
        chroma_out[i]=(uint8_t)(val<0?0:(val>255?255:val));
    }
}

/* auto-select best CCLM mode among several alpha candidates */
int wubu_cclm_best_mode(const uint8_t* luma_ref,const uint8_t* chroma_ref,
                          const uint8_t* actual_chroma,int bs,
                          double* out_alpha,double* out_beta){
    /* try multiple alpha values around the base estimate */
    static const double alphas[]={1.0,1.25,1.5,1.75,2.0};
    long best_sad=~(long)0;
    *out_alpha=1.0;*out_beta=128.0;
    
    for(int m=0;m<5;m++){
        /* compute beta so the model passes through the chroma mean */
        double sum_l=0,sum_c=0;
        for(int i=0;i<bs;i++){
            sum_l+=luma_ref[i];
            sum_c+=chroma_ref[i];
        }
        double mean_l=sum_l/bs,mean_c=sum_c/bs;
        double beta=mean_c-alphas[m]*mean_l;
        
        /* evaluate */
        long sad=0;
        for(int i=0;i<bs*bs;i++){
            int pred=(int)(alphas[m]*luma_ref[i]+beta);
            sad+=abs(actual_chroma[i]-pred);
        }
        
        if(sad<best_sad){
            best_sad=sad;
            *out_alpha=alphas[m];
            *out_beta=beta;
        }
    }
    return 0;
}

/* ===== G10.09: Joint Chroma Residual ===== */

/*
 * Check if Cb and Cr residuals can be jointly coded:
 * condition: sign(resCb) == sign(resCr) AND |resCb| ≥ T×|resCr| (T≈2)
 */
int wubu_jccr_check(const int16_t* res_cb,const int16_t* res_cr,
                      int n_coeffs,float threshold_ratio){
    for(int i=0;i<n_coeffs;i++){
        if(res_cb[i]==0&&res_cr[i]==0)continue;
        
        /* signs must match */
        int sign_cb=res_cb[i]>0?1:(res_cb[i]<0?-1:0);
        int sign_cr=res_cr[i]>0?1:(res_cr[i]<0?-1:0);
        if(sign_cb!=sign_cr)return 0;
        
        /* magnitude ratio must be within threshold */
        float abs_cb=(float)abs(res_cb[i]);
        float abs_cr=(float)abs(res_cr[i]);
        if(abs_cb==0&&abs_cr==0)continue;
        
        float larger=abs_cb>abs_cr?abs_cb:abs_cr;
        float smaller=abs_cb>abs_cr?abs_cr:abs_cb;
        if(larger/smaller>threshold_ratio)return 0;
    }
    return 1;
}

/* compute joint residual: resJoint = (resCb + resCr)/2 */
void wubu_jccr_compute(const int16_t* res_cb,const int16_t* res_cr,
                         int16_t* res_joint,int n_coeffs){
    for(int i=0;i<n_coeffs;i++)
        res_joint[i]=(res_cb[i]+res_cr[i])/2;
}

/* reconstruct both chroma channels from joint residual */
void wubu_jccr_split(const int16_t* res_joint,int16_t* res_cb,int16_t* res_cr,
                       int n_coeffs,int mode){
    /* mode 0: cb=cr=joint; mode 1: cb=joint, cr=-joint */
    for(int i=0;i<n_coeffs;i++){
        switch(mode){
            case 0:res_cb[i]=res_joint[i];res_cr[i]=res_joint[i];break;
            case 1:res_cb[i]=res_joint[i];res_cr[i]=-res_joint[i];break;
            default:res_cb[i]=res_joint[i];res_cr[i]=res_joint[i];
        }
    }
}
