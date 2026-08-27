/*
 * wubu_hamilton_encoder.c -- Hamilton Encoder (slermed from vhf_audio.py)
 *
 * Procedural version: generates quaternion latent space from RGB images
 * using coordinate-based encoding with positional encoding.
 * No trained weights — uses procedural transforms for slerm purposes.
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>

void wubu_hamilton_encoder_init(HamiltonEncoder* enc, int latent_grid_size, int d_model) {
    enc->latent_grid_size = latent_grid_size;
    enc->d_model = d_model;
}

void wubu_hamilton_encoder_free(HamiltonEncoder* enc) {
    (void)enc;
}

HamiltonKeys wubu_hamilton_encode_procedural(WubuRNG* rng, const float* images_rgb, int B, int img_size) {
    HamiltonKeys keys;
    keys.B = B;
    keys.H = img_size;
    keys.W = img_size;
    keys.context_dim = 3;

    int total_pixels = B * img_size * img_size;
    keys.quaternions = (float*)calloc((size_t)(total_pixels * 4), sizeof(float));
    keys.amplitude = (float*)calloc((size_t)total_pixels, sizeof(float));
    keys.context = (float*)calloc((size_t)(B * 3), sizeof(float));

    for (int b = 0; b < B; ++b) {
        const float* img = images_rgb + b * img_size * img_size * 3;

        /* Compute image statistics for context vector */
        float img_mean[3] = {0.0f, 0.0f, 0.0f};
        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                int idx = (y * img_size + x) * 3;
                img_mean[0] += img[idx + 0];
                img_mean[1] += img[idx + 1];
                img_mean[2] += img[idx + 2];
            }
        }
        float inv = 1.0f / (float)(img_size * img_size);
        keys.context[b * 3 + 0] = img_mean[0] * inv;
        keys.context[b * 3 + 1] = img_mean[1] * inv;
        keys.context[b * 3 + 2] = img_mean[2] * inv;

        /* Generate quaternion latent for each pixel */
        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                int pixel_idx = (b * img_size * img_size) + (y * img_size + x);
                int img_idx = (y * img_size + x) * 3;

                /* Procedural quaternion from pixel position + color */
                float u = (float)x / (float)(img_size - 1);
                float v = (float)y / (float)(img_size - 1);
                float r = img[img_idx + 0];
                float g = img[img_idx + 1];
                float b_col = img[img_idx + 2];

                /* Quaternion: rotation based on position, normalized */
                float qx = u * 2.0f - 1.0f;
                float qy = v * 2.0f - 1.0f;
                float qz = (r + g + b_col) / 3.0f;
                float qw = 1.0f;
                float norm = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);
                if (norm < 1e-8f) norm = 1.0f;

                keys.quaternions[pixel_idx * 4 + 0] = qx / norm;
                keys.quaternions[pixel_idx * 4 + 1] = qy / norm;
                keys.quaternions[pixel_idx * 4 + 2] = qz / norm;
                keys.quaternions[pixel_idx * 4 + 3] = qw / norm;

                /* Amplitude: based on luminance */
                keys.amplitude[pixel_idx] = wubu_rgb_to_grayscale((WubuRGB){r, g, b_col});
            }
        }
    }

    return keys;
}

void wubu_hamilton_keys_free(HamiltonKeys* keys) {
    free(keys->quaternions);
    free(keys->amplitude);
    free(keys->context);
    keys->quaternions = NULL;
    keys->amplitude = NULL;
    keys->context = NULL;
}
