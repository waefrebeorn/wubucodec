/*
 * wubu_av_fidelity.c -- audio-conditioned video fidelity loss (GAP-E005)
 *
 * The "resolution cheat" made into loss form: a video latent's fidelity
 * weight is modulated by the audio sideband energy at the matching sweep
 * segment. Loud/structured moments get tighter distortion budgets; silence
 * relaxes them. This is the loss the RL rate controller (env #2) optimizes.
 *
 *   w(t) = clamp( alpha + beta * band_energy_norm(t), w_min, 1 )
 *   L = mean_t( w(t) * D_video(t) )  +  lambda * R_total
 */

#include "wubu_av_fidelity.h"
#include <string.h>
#include <math.h>

void wubu_av_weights(const float* band_energy_norm,float* weights,
                     int T,float alpha,float beta,float w_min){
    for(int t=0;t<T;t++){
        /* mean across the 5 perceptual bands */
        float e=(band_energy_norm[(size_t)t*5+0]
                +band_energy_norm[(size_t)t*5+1]
                +band_energy_norm[(size_t)t*5+2]
                +band_energy_norm[(size_t)t*5+3]
                +band_energy_norm[(size_t)t*5+4])/5.0f;
        float w=alpha+beta*e;
        if(w<w_min)w=w_min; if(w>1.0f)w=1.0f;
        weights[t]=w;
    }
}

float wubu_av_fidelity_loss(const float* video_dist,const float* weights,
                            int T,float bits,float lambda){
    if(T<=0) return 0;
    double acc=0;
    for(int t=0;t<T;t++) acc+=weights[t]*(double)video_dist[t];
    return (float)(acc/T)+lambda*bits;
}
