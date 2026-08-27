/*
 * wubu_stft.c -- Short-Time Fourier Transform + inverse, C11 (GAP-E001)
 *
 * The Zephyr-HD "visual audio Kodak" needs a reversible spectral transform
 * in C. This is a radix-2 iterative FFT with Hann windowing and weighted
 * overlap-add ISTFT. Round-trip property: ISTFT(STFT(x)) == x for
 * hop = N/4 (75% overlap, Hann COLA-compliant) within float tolerance.
 */

#include "wubu_stft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------- iterative radix-2 FFT (in-place, split real/imag) ---- */
static void fft_radix2(float* re,float* im,int n,int inverse){
    /* bit reversal */
    for(int i=1,j=0;i<n;i++){
        int bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j){
            float t=re[i];re[i]=re[j];re[j]=t;
            t=im[i];im[i]=im[j];im[j]=t;
        }
    }
    for(int len=2;len<=n;len<<=1){
        float ang=(inverse?2.0f:-2.0f)*(float)M_PI/(float)len;
        float wr=cosf(ang),wi=sinf(ang);
        for(int i=0;i<n;i+=len){
            float cr=1.0f,ci=0.0f;
            for(int k=0;k<len/2;k++){
                int a=i+k,b=i+k+len/2;
                float tr=re[b]*cr-im[b]*ci;
                float ti=re[b]*ci+im[b]*cr;
                re[b]=re[a]-tr; im[b]=im[a]-ti;
                re[a]+=tr;      im[a]+=ti;
                float ncr=cr*wr-ci*wi;
                ci=cr*wi+ci*wr; cr=ncr;
            }
        }
    }
    if(inverse)
        for(int i=0;i<n;i++){re[i]/=(float)n;im[i]/=(float)n;}
}

int wubu_stft_init(WubuStft* st,int frame_size,int hop){
    if(frame_size<8||(frame_size&(frame_size-1))) return -1;  /* pow2 >=8 */
    if(hop<=0||hop>frame_size) return -2;
    memset(st,0,sizeof(*st));
    st->frame_size=frame_size;
    st->hop=hop;
    st->window=(float*)malloc(sizeof(float)*(size_t)frame_size);
    if(!st->window) return -3;
    /* periodic Hann: COLA-compliant at hop=N/4 */
    for(int i=0;i<frame_size;i++)
        st->window[i]=0.5f*(1.0f-cosf(2.0f*(float)M_PI*(float)i/(float)frame_size));
    return 0;
}

void wubu_stft_free(WubuStft* st){
    free(st->window);
    st->window=NULL;
}

int wubu_stft_num_frames(int signal_len,int frame_size,int hop){
    if(signal_len<frame_size) return 1;
    return 1+(signal_len-frame_size)/hop;
}

/* forward: x[len] -> frames[num_frames][F/2+1] complex pairs (re,im interleaved).
 * Real-input FFT: we pack via full complex FFT of the windowed frame
 * (simple, correct; optimization = later gap). */
float* wubu_stft_forward(const WubuStft* st,const float* x,int len,int* out_frames){
    int F=st->frame_size,hop=st->hop,Bins=F/2+1;
    int T=wubu_stft_num_frames(len,F,hop);
    float* out=malloc(sizeof(float)*(size_t)T*Bins*2);
    if(!out){*out_frames=0;return NULL;}

    float* re=malloc(sizeof(float)*(size_t)F);
    float* im=malloc(sizeof(float)*(size_t)F);
    for(int t=0;t<T;t++){
        int off=t*hop;
        for(int i=0;i<F;i++){
            float s=(off+i<len)?x[off+i]:0.0f;
            re[i]=s*st->window[i];
            im[i]=0.0f;
        }
        fft_radix2(re,im,F,0);
        for(int b=0;b<Bins;b++){
            out[((size_t)t*Bins+b)*2+0]=re[b];
            out[((size_t)t*Bins+b)*2+1]=im[b];
        }
    }
    free(re);free(im);
    *out_frames=T;
    return out;
}

/* inverse: weighted overlap-add. wsum[t] accumulates w^2; output /= wsum
 * where wsum > eps (edges). */
float* wubu_stft_inverse(const WubuStft* st,const float* frames,int num_frames,int out_len){
    int F=st->frame_size,hop=st->hop,Bins=F/2+1;
    float* y=calloc((size_t)out_len,sizeof(float));
    float* wsum=calloc((size_t)out_len,sizeof(float));
    float* re=malloc(sizeof(float)*(size_t)F);
    float* im=malloc(sizeof(float)*(size_t)F);
    if(!y||!wsum||!re||!im){free(y);free(wsum);free(re);free(im);return NULL;}

    for(int t=0;t<num_frames;t++){
        for(int b=0;b<Bins;b++){
            re[b]=frames[((size_t)t*Bins+b)*2+0];
            im[b]=frames[((size_t)t*Bins+b)*2+1];
        }
        /* reconstruct full spectrum by Hermitian symmetry */
        for(int b=Bins;b<F;b++){
            re[b]=re[F-b]; im[b]=-im[F-b];
        }
        fft_radix2(re,im,F,1);
        int off=t*hop;
        for(int i=0;i<F&&off+i<out_len;i++){
            float w=st->window[i];
            y[off+i]+=re[i]*w;
            wsum[off+i]+=w*w;
        }
    }
    for(int i=0;i<out_len;i++)
        if(wsum[i]>1e-8f) y[i]/=wsum[i];

    free(re);free(im);free(wsum);
    return y;
}
