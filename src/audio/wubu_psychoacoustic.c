/*
 * wubu_psychoacoustic.c -- GAP-E006: Psychoacoustic masking threshold
 *
 * Research source: Herre & Disch 2025 tutorial (arXiv:2504.16223) + MDPI
 * review (applsci-09-2854). Implements the core of a basic psychoacoustic
 * model:
 *   1. Bark-scale frequency mapping (Zwicker)
 *   2. Spreading function (+25 dB/Bark lower slope, -10 dB/Bark upper)
 *   3. Masking threshold per critical band = spread energy / alpha
 *   4. Signal-to-Mask Ratio per band -> drives bit allocation
 *
 * The SMR output feeds wubu_av_fidelity weights (GAP-E005) so that
 * perceptually important bands get tighter distortion budgets — this is
 * the "resolution cheat" applied to the audio sideband.
 */
#include "wubu_psychoacoustic.h"
#include <math.h>
#include <string.h>

/* Hz -> Bark scale (Zwicker approximation) */
float wubu_pa_hz_to_bark(float hz){
    return 13.0f*atanf(0.00076f*hz)+3.5f*atanf((hz/7500.0f)*(hz/7500.0f));
}

int wubu_pa_init(WubuPsychoacoustic* pa,int fft_size,int sample_rate){
    if(fft_size<64)return -1;
    pa->fft_size=fft_size;
    pa->sample_rate=sample_rate;
    int bins=fft_size/2+1;
    /* precompute bark index for each bin */
    pa->bin_bark=malloc(sizeof(float)*(size_t)bins);
    if(!pa->bin_bark)return -2;
    for(int k=0;k<bins;k++){
        float hz=(float)k*(float)sample_rate/(float)fft_size;
        pa->bin_bark[k]=wubu_pa_hz_to_bark(hz);
    }
    return 0;
}

void wubu_pa_free(WubuPsychoacoustic* pa){
    free(pa->bin_bark);pa->bin_bark=NULL;
}

/* compute masking threshold per STFT frame */
void wubu_pa_masking_threshold(const WubuPsychoacoustic* pa,
                                const float* spectrum, /* [bins] magnitudes */
                                float* threshold,      /* [bins] output */
                                float smr_db){         /* default 29 dB */
    int bins=pa->fft_size/2+1;
    float linear_smr=powf(10.0f,smr_db/20.0f);

    /* Step 1: compute spreading in bark domain.
     * For each bin k, energy from all bins j spreads with:
     *   lower slope +25 dB/Bark, upper slope -10 dB/Bark */
    for(int k=0;k<bins;k++){
        float masked_energy=0;
        float bk=pa->bin_bark[k];
        for(int j=0;j<bins;j++){
            float bj=pa->bin_bark[j];
            float db=bj-bk;
            /* spreading attenuation in dB */
            float atten;
            if(db>=0)
                atten=-10.0f*db;   /* upward spread loses 10 dB/Bark */
            else
                atten=+25.0f*(-db); /* downward spread gains 25 dB/Bark */
            /* convert to linear gain */
            float gain=powf(10.0f,atten/20.0f);
            if(gain>1.0f)gain=1.0f;   /* masking can't amplify */
            float mag=spectrum[j];
            masked_energy+=gain*mag*mag;
        }
        /* masking threshold = sqrt(masked_energy) / SMR_linear */
        threshold[k]=sqrtf(masked_energy)/linear_smr;
    }
}

/* Signal-to-Mask ratio per perceptual band (feeds AV fidelity weights) */
void wubu_pa_smr_per_band(const WubuPsychoacoustic* pa,
                          const float* spectrum,const float* threshold,
                          float* smr /* [5] */){
    int bins=pa->fft_size/2+1;
    /* use the same 5-band edges as wubu_bands */
    int edges[6]={0};
    float hi[5]={300.0f,4000.0f,10000.0f,16000.0f,1e9f};
    float bin_hz=(float)pa->sample_rate/(float)pa->fft_size;
    for(int b=0;b<5;b++){
        edges[b+1]=bins;
        for(int k=0;k<bins;k++)
            if(k*bin_hz>hi[b]){edges[b+1]=k;break;}
        if(edges[b+1]<=edges[b])edges[b+1]=edges[b]+1;
    }
    for(int b=0;b<5;b++){
        double sig_e=0,thr_e=0;
        for(int k=edges[b];k<edges[b+1]&&k<bins;k++){
            sig_e+=(double)spectrum[k]*spectrum[k];
            thr_e+=(double)threshold[k]*threshold[k];
        }
        if(thr_e<1e-12f)thr_e=1e-12f;
        smr[b]=(float)(sig_e/thr_e);
    }
}
