/* wubu_kodak.c -- Zephyr-HD Kodak: audio STFT <-> image layout
 *
 * The Kodak packs spectral data into an RGB image:
 *   R = log-magnitude (normalized)
 *   G = cos(phase) * 0.5 + 0.5
 *   B = sin(phase) * 0.5 + 0.5
 *
 * This allows using image compression tools for audio!
 */
#include "wubu_kodak.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int wubu_kodak_init(WubuKodak* kd, int width, int height) {
    kd->width = width;
    kd->height = height;
    kd->mag_scale = 1.0f;
    kd->pixels = calloc((size_t)width * height * 3, sizeof(float));
    return kd->pixels ? 0 : -1;
}

void wubu_kodak_free(WubuKodak* kd) {
    free(kd->pixels);
    kd->pixels = NULL;
}

/* Pack spectral frames into image layout
 * frames: [num_frames][bins*2] interleaved (re,im)
 * bins: FFT bins per frame
 *
 * Layout: image is width x height, each pixel holds one spectral bin.
 * Row-major: pixel (i,j) holds bin j of frame i (if i < num_frames).
 * If num_frames * bins > width * height, we tile/scroll.
 */
int wubu_kodak_pack(WubuKodak* kd, const float* frames, int num_frames, int bins) {
    if(!kd || !frames || !kd->pixels) return -1;
    
    int W = kd->width, H = kd->height;
    int total_pixels = W * H;
    int total_bins = num_frames * bins;
    
    /* Find max magnitude for normalization */
    float max_mag = 0.0f;
    for(int i = 0; i < total_bins; i++) {
        float re = frames[i * 2];
        float im = frames[i * 2 + 1];
        float mag = sqrtf(re * re + im * im);
        if(mag > max_mag) max_mag = mag;
    }
    
    /* Log-domain normalization */
    kd->mag_scale = max_mag;
    float log_max = logf(max_mag + 1e-12f);
    
    /* Pack into image */
    for(int p = 0; p < total_pixels; p++) {
        int bin_idx = p % total_bins;
        int frame = bin_idx / bins;
        int bin = bin_idx % bins;
        
        float re = frames[bin_idx * 2];
        float im = frames[bin_idx * 2 + 1];
        float mag = sqrtf(re * re + im * im);
        float phase = atan2f(im, re);
        
        /* R: log-magnitude normalized to [0,1] */
        kd->pixels[p * 3 + 0] = logf(mag + 1e-12f) / (log_max + 1e-12f);
        /* G: cos(phase) * 0.5 + 0.5 */
        kd->pixels[p * 3 + 1] = cosf(phase) * 0.5f + 0.5f;
        /* B: sin(phase) * 0.5 + 0.5 */
        kd->pixels[p * 3 + 2] = sinf(phase) * 0.5f + 0.5f;
    }
    
    return 0;
}

/* Unpack image back to spectral frames */
int wubu_kodak_unpack(const WubuKodak* kd, float* frames, int num_frames, int bins) {
    if(!kd || !frames || !kd->pixels) return -1;
    
    int W = kd->width, H = kd->height;
    int total_pixels = W * H;
    int total_bins = num_frames * bins;
    
    float log_max = logf(kd->mag_scale + 1e-12f);
    
    /* Unpack from image */
    for(int p = 0; p < total_pixels && p < total_bins; p++) {
        /* R: log-magnitude */
        float log_mag = kd->pixels[p * 3 + 0] * log_max;
        float mag = expf(log_mag);
        /* G, B: phase */
        float cos_phase = kd->pixels[p * 3 + 1] * 2.0f - 1.0f;
        float sin_phase = kd->pixels[p * 3 + 2] * 2.0f - 1.0f;
        float phase = atan2f(sin_phase, cos_phase);
        
        frames[p * 2 + 0] = mag * cosf(phase);
        frames[p * 2 + 1] = mag * sinf(phase);
    }
    
    return 0;
}
