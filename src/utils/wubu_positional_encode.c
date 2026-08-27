/*
 * wubu_positional_encode.c -- Positional Encoding (slermed from vhf_audio.py)
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>

float* wubu_positional_encode(const float* x, int D, int num_freqs) {
    int out_dim = D + 2 * num_freqs * D;
    float* out = (float*)malloc((size_t)out_dim * sizeof(float));
    if (!out) return NULL;

    /* Copy original */
    memcpy(out, x, (size_t)D * sizeof(float));

    /* Compute frequency bands: 2^k * PI for k in [0, num_freqs) */
    for (int k = 0; k < num_freqs; ++k) {
        float freq = powf(2.0f, (float)k) * (float)M_PI;
        for (int d = 0; d < D; ++d) {
            float val = x[d] * freq;
            out[D + k * 2 * D + d] = sinf(val);
            out[D + k * 2 * D + D + d] = cosf(val);
        }
    }
    return out;
}
