/*
 * wubu_bands.c -- Perceptual-band spectral split (GAP-E002)
 *
 * The Zephyr-HD Kodak's 5-band layout (Bass/Mids/Presence/Treble/Harmonics)
 * as reusable C: partitions STFT bins into perceptual bands by frequency,
 * computes per-band energy, and L2-normalizes so band TEXTURE is separable
 * from band LOUDNESS — the separation that makes audio a trainable image.
 */

#include "wubu_bands.h"
#include <string.h>
#include <math.h>

int wubu_band_split(const WubuStft* st, WubuBandTable* bt){
    if(!st->window||st->frame_size<64) return -1;
    int bins=st->frame_size/2+1;
    float bin_hz=48000.0f/(float)st->frame_size;  /* assume 48k; doc'd */
    int edges[6]={0};
    const float hi[5]={300.0f,4000.0f,10000.0f,16000.0f,1e9f};
    for(int b=0;b<5;b++){
        edges[b+1]=bins;
        for(int k=0;k<bins;k++)
            if(k*bin_hz>hi[b]){edges[b+1]=k;break;}
        if(edges[b+1]<=edges[b]) edges[b+1]=edges[b]+1;
    }
    for(int b=0;b<5;b++){
        bt->start[b]=edges[b];
        bt->end[b]=edges[b+1]>bins?bins:edges[b+1];
    }
    return 0;
}

/* per-band RMS energy into out[5] */
void wubu_band_energy(const WubuBandTable* bt,const float* frames,
                      int num_frames,int bins,float* out){
    for(int b=0;b<5;b++){
        double acc=0; long cnt=0;
        for(int t=0;t<num_frames;t++)
            for(int k=bt->start[b];k<bt->end[b]&&k<bins;k++){
                float re=frames[((size_t)t*bins+k)*2];
                float im=frames[((size_t)t*bins+k)*2+1];
                acc+=(double)re*re+im*im;
                cnt++;
            }
        out[b]=sqrtf((float)(acc/(cnt?cnt:1)));
    }
}

/* normalize band energies to unit max: texture-preserving loudness removal */
void wubu_band_normalize(float* e5){
    float mx=1e-9f;
    for(int b=0;b<5;b++) if(e5[b]>mx) mx=e5[b];
    for(int b=0;b<5;b++) e5[b]/=mx;
}
