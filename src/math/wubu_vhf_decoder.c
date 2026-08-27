/*
 * wubu_vhf_decoder.c -- VHF Decoder (slermed from vhf_audio.py)
 *
 * Procedural version: decodes Hamilton keys + context + coordinates
 * into RGB output using coordinate-based sampling with positional encoding.
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>

void wubu_vhf_decoder_init(VHFDecoder* dec, int d_model) {
    dec->d_model = d_model;
}

void wubu_vhf_decoder_free(VHFDecoder* dec) {
    (void)dec;
}

float* wubu_vhf_decode_procedural(WubuRNG* rng, const HamiltonKeys* keys,
                                   const float* coords, int N) {
    (void)rng; /* procedural decode doesn't use RNG */
    float* output = (float*)malloc((size_t)(N * 3) * sizeof(float));
    if (!output) return NULL;

    int B = keys->B;
    int H = keys->H;
    int W = keys->W;

    for (int i = 0; i < N; ++i) {
        /* Get coordinate in [-1,1] */
        float cx = coords[i * 2 + 0];
        float cy = coords[i * 2 + 1];

        /* Convert to pixel space */
        float xf = (cx + 1.0f) / 2.0f * (float)(W - 1);
        float yf = (cy + 1.0f) / 2.0f * (float)(H - 1);

        /* Bilinear sample from each batch entry (use first) */
        int x0 = (int)floorf(xf);
        int y0 = (int)floorf(yf);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 >= W) x1 = W - 1;
        if (y1 >= H) y1 = H - 1;

        float wx = xf - (float)x0;
        float wy = yf - (float)y0;

        /* Sample quaternion + amplitude from first batch entry */
        int base_idx = y0 * W + x0;
        float q00[4] = {keys->quaternions[base_idx*4+0], keys->quaternions[base_idx*4+1],
                        keys->quaternions[base_idx*4+2], keys->quaternions[base_idx*4+3]};
        float a00 = keys->amplitude[base_idx];

        base_idx = y0 * W + x1;
        float q01[4] = {keys->quaternions[base_idx*4+0], keys->quaternions[base_idx*4+1],
                        keys->quaternions[base_idx*4+2], keys->quaternions[base_idx*4+3]};
        float a01 = keys->amplitude[base_idx];

        base_idx = y1 * W + x0;
        float q10[4] = {keys->quaternions[base_idx*4+0], keys->quaternions[base_idx*4+1],
                        keys->quaternions[base_idx*4+2], keys->quaternions[base_idx*4+3]};
        float a10 = keys->amplitude[base_idx];

        base_idx = y1 * W + x1;
        float q11[4] = {keys->quaternions[base_idx*4+0], keys->quaternions[base_idx*4+1],
                        keys->quaternions[base_idx*4+2], keys->quaternions[base_idx*4+3]};
        float a11 = keys->amplitude[base_idx];

        /* Bilinear interpolation of quaternion */
        float w00 = (1-wx)*(1-wy), w01 = (1-wx)*wy, w10 = wx*(1-wy), w11 = wx*wy;
        float q[4];
        for (int c = 0; c < 4; ++c) {
            float v00 = (c==0?q[0]:(c==1?q[1]:(c==2?q[2]:q[3]))); /* placeholder */
            q[c] = w00*q00[c] + w01*q01[c] + w10*q10[c] + w11*q11[c];
        }
        float amp = w00*a00 + w01*a01 + w10*a10 + w11*a11;

        /* Decode quaternion + amplitude into RGB */
        /* Simple procedural decode: use quaternion rotation on a base color */
        float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm < 1e-8f) norm = 1.0f;
        float qx = q[0]/norm, qy = q[1]/norm, qz = q[2]/norm, qw = q[3]/norm;

        /* Rotation of [1,0,0] by quaternion -> RGB */
        float rx = 1.0f - 2.0f*(qy*qy + qz*qz);
        float ry = 2.0f*(qx*qy + qw*qz);
        float rz = 2.0f*(qx*qz - qw*qy);

        /* Apply amplitude and context modulation */
        float ctx[3] = {keys->context[0], keys->context[1], keys->context[2]};
        output[i * 3 + 0] = rx * amp * ctx[0];
        output[i * 3 + 1] = ry * amp * ctx[1];
        output[i * 3 + 2] = rz * amp * ctx[2];

        /* Clamp to [-1,1] */
        if (output[i*3+0] > 1.0f) output[i*3+0] = 1.0f;
        if (output[i*3+0] < -1.0f) output[i*3+0] = -1.0f;
        if (output[i*3+1] > 1.0f) output[i*3+1] = 1.0f;
        if (output[i*3+1] < -1.0f) output[i*3+1] = -1.0f;
        if (output[i*3+2] > 1.0f) output[i*3+2] = 1.0f;
        if (output[i*3+2] < -1.0f) output[i*3+2] = -1.0f;
    }

    return output;
}
