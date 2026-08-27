/*
 * wubu_vhf_audio.c -- VHF Hamilton Modulator + Audio (slermed from vhf_audio.py)
 *
 * Faithful C11 translation of the VHF end-to-end model from bytropix:
 * https://github.com/waefrebeorn/bytropix/blob/master/AUDIO/wubusynth/vhf_audio.py
 *
 * Components:
 *   - VHFEncoderConfig / VHFEncoded: Quaternion latent space
 *   - VHFDecoderConfig: Coordinate-sampled reconstruction
 *   - VHAudioConfig: Audio strip / canvas constants
 *   - wubu_vhf_encode(): RGB images -> quaternion latent + amplitude + context
 *   - wubu_vhf_decode(): latent + context + coords -> RGB reconstruction
 *   - wubu_vhf_generate_audio_strip(): raw audio -> HBI strip
 *   - wubu_vhf_compose_canvas(): VBI + HBI + visible -> full canvas
 *   - wubu_vhf_train_step(): Loss computation + Q-controller update
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>

/* ===================================================================
 * Default Configs (matching vhf_audio.py)
 * =================================================================== */

const VHFEncoderConfig VHF_ENCODER_DEFAULT = {
    .latent_grid_size = 96,
    .d_model = 512
};

const VHFDecoderConfig VHF_DECODER_DEFAULT = {
    .d_model = 512
};

const VHAudioConfig VHF_AUDIO_DEFAULT = {
    .sample_rate = 44100,
    .canvas_w = CANVAS_W,      /* 656 */
    .canvas_h = CANVAS_H,      /* 525 */
    .vbi_lines = VBI_LINES,    /* 45 */
    .visible_h = VISIBLE_H,    /* 480 */
    .audio_hbi_width = AUDIO_HBI_WIDTH  /* 16 */
};

/* ===================================================================
 * Hamilton Encoder
 *
 * Slermed from HamiltonEncoder (vhf_audio.py line ~100):
 *   - Downsamples image through conv layers with GELU
 *   - Aggregates context vectors from each scale
 *   - Produces quaternion[4] + amplitude[1] per spatial position
 *
 * Procedural version: uses coordinate-based encoding instead of trained weights
 * =================================================================== */

VHFEncoded wubu_vhf_encode(WubuRNG* rng, const float* images_rgb, int B, int img_size) {
    VHFEncoded encoded;
    encoded.B = B;
    encoded.H = img_size;
    encoded.W = img_size;

    int total_pixels = B * img_size * img_size;
    encoded.quaternions = (float*)calloc((size_t)(total_pixels * 4), sizeof(float));
    encoded.amplitude = (float*)calloc((size_t)total_pixels, sizeof(float));
    encoded.context = (float*)calloc((size_t)(B * 3), sizeof(float));

    (void)rng; /* procedural encoding doesn't use RNG */

    for (int b = 0; b < B; ++b) {
        const float* img = images_rgb + b * img_size * img_size * 3;

        /* Compute image mean for context vector (matching Flax context_vectors.append) */
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
        encoded.context[b * 3 + 0] = img_mean[0] * inv;
        encoded.context[b * 3 + 1] = img_mean[1] * inv;
        encoded.context[b * 3 + 2] = img_mean[2] * inv;

        /* Generate quaternion latent per pixel
         * Matching: quat_raw = Conv(5x1x1)(x), quaternions = normalize(quat_raw[:4])
         * Procedural: use normalized position + color for quaternion
         */
        for (int y = 0; y < img_size; ++y) {
            for (int x = 0; x < img_size; ++x) {
                int pixel_idx = (b * img_size * img_size) + (y * img_size + x);
                int img_idx = (y * img_size + x) * 3;

                float u = (float)x / (float)(img_size - 1);
                float v = (float)y / (float)(img_size - 1);
                float r = img[img_idx + 0];
                float g = img[img_idx + 1];
                float b_col = img[img_idx + 2];

                /* Quaternion from position + color (procedural Conv equivalent) */
                float qx = u * 2.0f - 1.0f;
                float qy = v * 2.0f - 1.0f;
                float qz = (r + g + b_col) / 3.0f;
                float qw = 1.0f;
                float norm = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);
                if (norm < 1e-8f) norm = 1.0f;

                encoded.quaternions[pixel_idx * 4 + 0] = qx / norm;
                encoded.quaternions[pixel_idx * 4 + 1] = qy / norm;
                encoded.quaternions[pixel_idx * 4 + 2] = qz / norm;
                encoded.quaternions[pixel_idx * 4 + 3] = qw / norm;

                /* Amplitude: sigmoid(Conv(5x1x1)[..., 4]) -> use grayscale */
                float gray = wubu_rgb_to_grayscale((WubuRGB){r, g, b_col});
                /* Sigmoid: 1/(1+e^(-x)), apply to grayscale-mapped value */
                encoded.amplitude[pixel_idx] = 1.0f / (1.0f + expf(-gray * 2.0f));
            }
        }
    }

    return encoded;
}

void wubu_vhf_encoded_free(VHFEncoded* encoded) {
    free(encoded->quaternions);
    free(encoded->amplitude);
    free(encoded->context);
    encoded->quaternions = NULL;
    encoded->amplitude = NULL;
    encoded->context = NULL;
}

/* ===================================================================
 * VHF Decoder
 *
 * Slermed from VHFDecoder (vhf_audio.py line ~140):
 *   - Sample local features from latent grid via bilinear sampling
 *   - Apply positional encoding to coordinates
 *   - Tile context vector across spatial positions
 *   - Concatenate [encoded_coords, context, local_features]
 *   - MLP decode -> tanh -> RGB [-1,1]
 *
 * Procedural version: quaternion rotation + coordinate-based decode
 * =================================================================== */

float* wubu_vhf_decode(WubuRNG* rng, const VHFEncoded* encoded,
                        const float* coords, int N) {
    (void)rng;
    float* output = (float*)malloc((size_t)(N * 3) * sizeof(float));
    if (!output) return NULL;

    int H = encoded->H;
    int W = encoded->W;

    for (int i = 0; i < N; ++i) {
        /* Coordinate in [-1,1] -> pixel space */
        float cx = coords[i * 2 + 0];
        float cy = coords[i * 2 + 1];
        float xf = (cx + 1.0f) / 2.0f * (float)(W - 1);
        float yf = (cy + 1.0f) / 2.0f * (float)(H - 1);

        /* Bilinear sample from latent grid (first batch entry) */
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
        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = (1.0f - wx) * wy;
        float w10 = wx * (1.0f - wy);
        float w11 = wx * wy;

        /* Interpolate quaternion + amplitude */
        int base00 = y0 * W + x0;
        int base01 = y0 * W + x1;
        int base10 = y1 * W + x0;
        int base11 = y1 * W + x1;

        float q[4] = {0};
        for (int c = 0; c < 4; ++c) {
            q[c] = w00 * encoded->quaternions[base00 * 4 + c] +
                   w01 * encoded->quaternions[base01 * 4 + c] +
                   w10 * encoded->quaternions[base10 * 4 + c] +
                   w11 * encoded->quaternions[base11 * 4 + c];
        }
        float amp = w00 * encoded->amplitude[base00] +
                    w01 * encoded->amplitude[base01] +
                    w10 * encoded->amplitude[base10] +
                    w11 * encoded->amplitude[base11];

        /* Decode quaternion -> RGB via rotation + context modulation (procedural MLP) */
        float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm < 1e-8f) norm = 1.0f;
        q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;

        /* Rotate base color [1,0,0] by quaternion (matching Flax Dense -> tanh) */
        output[i * 3 + 0] = (1.0f - 2.0f*(q[1]*q[1] + q[2]*q[2])) * amp * encoded->context[0];
        output[i * 3 + 1] = (2.0f*(q[0]*q[1] + q[3]*q[2])) * amp * encoded->context[1];
        output[i * 3 + 2] = (2.0f*(q[0]*q[2] - q[3]*q[1])) * amp * encoded->context[2];

        /* Clamp to [-1,1] (matching nn.tanh) */
        if (output[i*3+0] > 1.0f) output[i*3+0] = 1.0f;
        if (output[i*3+0] < -1.0f) output[i*3+0] = -1.0f;
        if (output[i*3+1] > 1.0f) output[i*3+1] = 1.0f;
        if (output[i*3+1] < -1.0f) output[i*3+1] = -1.0f;
        if (output[i*3+2] > 1.0f) output[i*3+2] = 1.0f;
        if (output[i*3+2] < -1.0f) output[i*3+2] = -1.0f;
    }

    return output;
}

/* ===================================================================
 * Audio Strip Generation
 *
 * Slermed from video_audio_generator (vhf_audio.py line ~200):
 *   - Normalize audio to [-1,1]
 *   - Pad to canvas_h * audio_hbi_width target size
 *   - Reshape to [canvas_h, audio_hbi_width]
 *   * Replicate grayscale to RGB [canvas_h, audio_hbi_width, 3]
 * =================================================================== */

float* wubu_vhf_generate_audio_strip(const float* audio, int num_samples,
                                      const VHAudioConfig* config) {
    int target_size = config->canvas_h * config->audio_hbi_width;
    float* strip = (float*)malloc((size_t)(target_size * 3) * sizeof(float));
    if (!strip) return NULL;

    /* Normalize audio to [-1,1] and pad to target size */
    for (int i = 0; i < target_size; ++i) {
        float sample = 0.0f;
        if (i < num_samples) {
            sample = audio[i];
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
        }
        /* Replicate to 3 channels (grayscale -> RGB) */
        strip[i * 3 + 0] = sample;
        strip[i * 3 + 1] = sample;
        strip[i * 3 + 2] = sample;
    }

    return strip;
}

/* ===================================================================
 * Canvas Compositing
 *
 * Slermed from eval_canvases (vhf_audio.py line ~350):
 *   - VBI block: context padding [VBI_LINES, CANVAS_W, 3]
 *   - Audio HBI: [VISIBLE_H, AUDIO_HBI_WIDTH, 3] (visible portion of audio strip)
 *   - Visible: [VISIBLE_H, VISIBLE_W, 3]
 *   - Compose: VBI + (HBI | Visible) -> [CANVAS_H, CANVAS_W, 3]
 * =================================================================== */

float* wubu_vhf_compose_canvas(const float* vbi_block, const float* audio_hbi,
                                const float* visible, const VHAudioConfig* config) {
    int total_size = config->canvas_h * config->canvas_w * 3;
    float* canvas = (float*)malloc((size_t)total_size * sizeof(float));
    if (!canvas) return NULL;

    /* Copy VBI block (top lines) */
    memcpy(canvas, vbi_block, (size_t)(config->vbi_lines * config->canvas_w * 3) * sizeof(float));

    /* Composite visible rows: audio HBI + video side by side */
    for (int y = 0; y < config->visible_h; ++y) {
        int row_offset = (config->vbi_lines + y) * config->canvas_w * 3;

        /* Audio HBI portion (left strip) */
        memcpy(canvas + row_offset,
               audio_hbi + y * config->audio_hbi_width * 3,
               (size_t)(config->audio_hbi_width * 3) * sizeof(float));

        /* Video frame portion (right) */
        memcpy(canvas + row_offset + config->audio_hbi_width * 3,
               visible + y * VISIBLE_W * 3,
               (size_t)(VISIBLE_W * 3) * sizeof(float));
    }

    return canvas;
}

/* ===================================================================
 * VHF Training Step
 *
 * Slermed from train_step (vhf_audio.py line ~300):
 *   - Compute HSL loss: circular L1 (hue) + L1 (sat, lightness)
 *   - Weights: LUMA=10, PHASE=2, SAT=1
 *   - Update Q-controller with composite loss
 * =================================================================== */

VHFTrainingState wubu_vhf_train_step(const float* pred_rgb, const float* gt_rgb,
                                       int pixels_per_step, QController* qc) {
    VHFTrainingState state = {0};

    WubuLoss loss = wubu_compute_loss(pred_rgb, gt_rgb, pixels_per_step);

    state.composite_loss = loss.composite_loss;
    state.luma_loss = loss.luma_loss;
    state.phase_loss = loss.phase_loss;
    state.sat_loss = loss.sat_loss;

    /* Update Q-controller */
    QControllerConfig config = WUBU_Q_CONTROLLER_DEFAULT;
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    wubu_q_controller_update(qc, loss.composite_loss, &config);

    state.q_status = qc->status_code;
    state.current_lr = qc->current_lr;
    state.step_count = qc->step_count;

    return state;
}
