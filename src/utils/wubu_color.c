/*
 * wubu_color.c -- GROUP 17: Color space pipeline
 *
 * G17.01: BT.601 ↔ BT.709 conversion matrices
 * G17.02: BT.2020 for HDR/WCG
 * G17.06: HDR→SDR tone mapping (Reinhard)
 */
#define M_PI 3.14159265358979f
#include "wubu_color.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Conversion matrices ===== */

typedef struct {
    /* Y = m[0]*R + m[1]*G + m[2]*B */
    double kr,kg,kb;
} ColorMatrix;

static const ColorMatrix cm_601 ={0.299,  0.587,  0.114};
static const ColorMatrix cm_709 ={0.2126, 0.7152, 0.0722};
static const ColorMatrix cm_2020={0.2627, 0.6780, 0.0593};

/* RGB→YUV using specified matrix */
void wubu_rgb_to_yuv_cm(const uint8_t* rgb,uint8_t* y,uint8_t* cb,uint8_t* cr,
                          long n_pixels,wubu_color_std_t std){
    const ColorMatrix* cm;
    switch(std){
        case WUBU_CS_BT601: cm=&cm_601;break;
        case WUBU_CS_BT709: cm=&cm_709;break;
        case WUBU_CS_BT2020:cm=&cm_2020;break;
        default: cm=&cm_601;
    }
    
    for(long i=0,j=0;j<n_pixels;i+=3,j++){
        double R=rgb[i],G=rgb[i+1],B=rgb[i+2];
        double Y=cm->kr*R+cm->kg*G+cm->kb*B;
        y[j]=(uint8_t)(Y<0?0:(Y>255?255:Y));
    }
}

/* ===== BT.709 full conversion (with chroma) ===== */

void wubu_rgb_to_709(const uint8_t* rgb,uint8_t* y,uint8_t* u,uint8_t* v,
                      long n_pixels){
    for(long i=0,j=0;j<n_pixels;i+=3,j++){
        double R=rgb[i]/255.0,G=rgb[i+1]/255.0,B=rgb[i+2]/255.0;
        
        double Y=0.2126*R+0.7152*G+0.0722*B;
        double Cb=-0.1146*R-0.3854*G+0.5*B;
        double Cr=0.5*R-0.4542*G-0.0458*B;
        
        y[j]=(uint8_t)((Y*255.0+0.5));
        u[j]=(uint8_t)((Cb*127.0+128.0+0.5));
        v[j]=(uint8_t)((Cr*127.0+128.0+0.5));
    }
}

/* ===== BT.2020 for HDR ===== */

void wubu_rgb_to_2020(const uint8_t* rgb,uint16_t* y,uint16_t* u,uint16_t* v,
                       long n_pixels){
    for(long i=0,j=0;j<n_pixels;i+=3,j++){
        double R=rgb[i]/255.0,G=rgb[i+1]/255.0,B=rgb[i+2]/255.0;
        
        double Y=0.2627*R+0.6780*G+0.0593*B;
        double Cb=(B-Y)/1.8814;
        double Cr=(R-Y)/1.4746;
        
        /* output as 10-bit for HDR */
        y[j]=(uint16_t)(Y*1023.0+0.5);
        u[j]=(uint16_t)(Cb*511.0+512.0+0.5);
        v[j]=(uint16_t)(Cr*511.0+512.0+0.5);
    }
}

/* ===== G17.06: Reinhard tone mapping ===== */

/* Simple global tone mapping: maps [0,max_luma] to [0,255] with knee curve */
void wubu_tonemap_reinhard(const float* hdr_rgb,uint8_t* sdr_rgb,
                             long n_pixels,float max_luminance){
    for(long i=0;i<n_pixels;i++){
        /* compute luminance */
        float L=0.2126f*hdr_rgb[i*3]+0.7152f*hdr_rgb[i*3+1]+0.0722f*hdr_rgb[i*3+2];
        
        /* Reinhard: L_out = L / (1 + L/L_white) */
        float L_white=max_luminance*0.8f; /* white point at 80% of max */
        float L_mapped=L/(1.0f+L/L_white);
        
        /* normalize to [0,1] then scale to [0,255] */
        float scale=L_mapped/(L>0.001f?L:0.001f); /* ratio preserves color ratios */
        
        for(int c=0;c<3;c++){
            float val=hdr_rgb[i*3+c]*scale;
            if(val>255)val=255;
            if(val<0)val=0;
            sdr_rgb[i*3+c]=(uint8_t)(val+0.5f);
        }
    }
}

/* PQ (SMPTE ST 2084) EOTF approximation for testing */
float wubu_pq_eotf(uint16_t pq_value){
    /* simplified PQ inverse EOTF for luminance in nits */
    double x=(double)pq_value/1023.0;
    /* approximate: L ≈ 10000 * x^4 for quick test purposes */
    return (float)(10000.0*pow(x,4.0));
}

/* float-based color utilities (types from wubumath.h) */
#include <wubumath.h>

static float hue_to_rgb(float p,float q,float t){
    if(t<0)t+=1;if(t>1)t-=1;
    if(t<1/6.0)return p+(q-p)*6*t;
    if(t<1/2.0)return q;
    if(t<2/3.0)return p+(q-p)*(2/3.0-t)*6;
    return p;
}

WubuHSL wubu_rgb_to_hsl(WubuRGB rgb){
    /* rgb values are 0.0-1.0 */
    float mx=(rgb.r>rgb.g&&rgb.r>rgb.b)?rgb.r:(rgb.g>rgb.b)?rgb.g:rgb.b;
    float mn=(rgb.r<rgb.g&&rgb.r<rgb.b)?rgb.r:(rgb.g<rgb.b)?rgb.g:rgb.b;
    float l=(mx+mn)/2;
    float s=0,h=0;
    if(mx!=mn){
        float d=mx-mn;
        s=l>0.5f?d/(2-mx-mn):d/(mx+mn);
        if(mx==rgb.r)h=(rgb.g-rgb.b)/d+(rgb.g<rgb.b?6:0);
        else if(mx==rgb.g)h=(rgb.b-rgb.r)/d+2;
        else h=(rgb.r-rgb.g)/d+4;
        h/=6;
    }
    WubuHSL out={h,s,l};
    return out;
}

float wubu_circular_l1_loss(float pred,float target){
    /* values on [0,1] circle: shortest distance around the ring */
    float diff=fmodf(pred-target+1.5f,1.0f)-0.5f;
    if(diff<0)diff=-diff; /* abs */
    if(diff>0.5f)diff=1.0f-diff;
    return diff;
}

WubuRGB wubu_hsl_to_rgb(WubuHSL hsl){
    float h=hsl.h,s=hsl.s,l=hsl.l;
    float r,g,b;
    if(s==0){r=g=b=l;}
    else{
        float q=l<0.5f?l*(1+s):l+s-l*s;
        float p=2*l-q;
        r=hue_to_rgb(p,q,h+1.0f/3);
        g=hue_to_rgb(p,q,h);
        b=hue_to_rgb(p,q,h-1.0f/3);
    }
    WubuRGB out={r,g,b};
    return out;
}

float wubu_rgb_to_grayscale(WubuRGB rgb){
    return 0.299f*rgb.r+0.587f*rgb.g+0.114f*rgb.b;
}
