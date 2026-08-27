/*
 * wubu_utils.c -- Utility functions (bilinear sampling, box blur)
 * Slermed from vhf_audio.py
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>

float* wubu_bilinear_sample(const float* image, int H, int W, int C,
                            const float* coords, int N) {
    float* output = (float*)malloc((size_t)(N * C) * sizeof(float));
    if (!output) return NULL;

    for (int i = 0; i < N; ++i) {
        /* Convert from [-1,1] to pixel coords */
        float xf = (coords[i * 2 + 0] + 1.0f) / 2.0f * (float)(W - 1);
        float yf = (coords[i * 2 + 1] + 1.0f) / 2.0f * (float)(H - 1);

        int x0 = (int)floorf(xf);
        int y0 = (int)floorf(yf);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = xf - (float)x0;
        float wy = yf - (float)y0;

        if (x0 < 0) { x0 = 0; x1 = 0; wx = 0.0f; }
        if (y0 < 0) { y0 = 0; y1 = 0; wy = 0.0f; }
        if (x1 >= W) { x1 = W - 1; x0 = W - 1; wx = 0.0f; }
        if (y1 >= H) { y1 = H - 1; y0 = H - 1; wy = 0.0f; }

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = (1.0f - wx) * wy;
        float w10 = wx * (1.0f - wy);
        float w11 = wx * wy;

        for (int c = 0; c < C; ++c) {
            float v00 = image[(y0 * W + x0) * C + c];
            float v01 = image[(y1 * W + x0) * C + c];
            float v10 = image[(y0 * W + x1) * C + c];
            float v11 = image[(y1 * W + x1) * C + c];
            output[i * C + c] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        }
    }
    return output;
}

float* wubu_box_blur_5x5(const float* image, int H, int W) {
    float* output = (float*)malloc((size_t)(H * W) * sizeof(float));
    float* temp = (float*)malloc((size_t)(H * W) * sizeof(float));
    if (!output || !temp) { free(output); free(temp); return NULL; }

    /* Horizontal pass */
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                int xk = x + k;
                if (xk < 0) xk = 0;
                if (xk >= W) xk = W - 1;
                sum += image[y * W + xk];
            }
            temp[y * W + x] = sum / 5.0f;
        }
    }

    /* Vertical pass */
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int k = -2; k <= 2; ++k) {
                int yk = y + k;
                if (yk < 0) yk = 0;
                if (yk >= H) yk = H - 1;
                sum += temp[yk * W + x];
            }
            output[y * W + x] = sum / 5.0f;
        }
    }

    free(temp);
    return output;
}

float* wubu_generate_audio_strip(const float* audio_samples, int num_samples) {
    /* Generate HBI audio strip from raw audio samples
     * Target size: CANVAS_H * AUDIO_HBI_WIDTH floats
     * Output: [CANVAS_H * AUDIO_HBI_WIDTH * 3] RGB (grayscale replicated)
     */
    int target_size = CANVAS_H * AUDIO_HBI_WIDTH;
    float* strip = (float*)malloc((size_t)(target_size * 3) * sizeof(float));
    if (!strip) return NULL;

    /* Normalize and pad audio to target size */
    for (int i = 0; i < CANVAS_H * AUDIO_HBI_WIDTH; ++i) {
        float sample = 0.0f;
        if (i < num_samples) {
            sample = audio_samples[i];
            /* Clamp to [-1,1] */
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
        }
        /* Replicate to 3 channels */
        strip[i * 3 + 0] = sample;
        strip[i * 3 + 1] = sample;
        strip[i * 3 + 2] = sample;
    }
    return strip;
}

float* wubu_compose_canvas(const float* vbi_block, const float* audio_hbi, const float* visible) {
    /* Composite a full canvas:
     * - VBI_LINES rows: vbi_block [* CANVAS_W * 3]
     * - VISIBLE_H rows: [audio_hbi[* AUDIO_HBI_WIDTH * 3] + visible[* VISIBLE_W * 3]]
     * Total: CANVAS_H * CANVAS_W * 3
     */
    int total_size = CANVAS_H * CANVAS_W * 3;
    float* canvas = (float*)malloc((size_t)total_size * sizeof(float));
    if (!canvas) return NULL;

    /* Copy VBI block */
    memcpy(canvas, vbi_block, (size_t)(VBI_LINES * CANVAS_W * 3) * sizeof(float));

    /* Composite visible rows */
    for (int y = 0; y < VISIBLE_H; ++y) {
        int row_offset = (VBI_LINES + y) * CANVAS_W * 3;
        /* HBI audio portion */
        memcpy(canvas + row_offset,
               audio_hbi + y * AUDIO_HBI_WIDTH * 3,
               (size_t)(AUDIO_HBI_WIDTH * 3) * sizeof(float));
        /* Visible video frame */
        memcpy(canvas + row_offset + AUDIO_HBI_WIDTH * 3,
               visible + y * VISIBLE_W * 3,
               (size_t)(VISIBLE_W * 3) * sizeof(float));
    }

    return canvas;
}
