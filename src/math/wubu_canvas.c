/*
 * wubu_canvas.c -- Multi-resolution WuBu Canvas System
 *
 * Implements the full VHF pipeline at configurable resolutions.
 * Faithful extension of the existing Hamilton encoder/decoder/audio pipeline.
 */

#include "wubu_canvas.h"
#include "wubumath.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================================================================
 * Resolution Table
 * =================================================================== */

static const WubuCanvasConfig g_resolutions[WUBU_RES_COUNT] = {
    /* 360P */  { 640,  360, 45, 16, 656,  405,  656*405,  "360P"  },
    /* 480P */  { 854,  480, 45, 16, 870,  525,  870*525,  "480P"  },
    /* 720P */  {1280,  720, 45, 16,1296,  765, 1296*765,  "720P"  },
    /* 1080P */ {1920, 1080, 45, 16,1936, 1125, 1936*1125,"1080P" },
    /* 1440P */ {2560, 1440, 45, 16,2576, 1485, 2576*1485,"1440P" },
    /* 4K */    {3840, 2160, 45, 16,3856, 2205, 3856*2205,"4K"    },
};

const WubuCanvasConfig* wubu_canvas_get_config(WubuResolution res) {
    if (res < 0 || res >= WUBU_RES_COUNT) return &g_resolutions[WUBU_RES_480P];
    return &g_resolutions[res];
}

WubuCanvasConfig wubu_canvas_make_config(int visible_w, int visible_h,
                                          int vbi_lines, int hbi_width) {
    WubuCanvasConfig cfg;
    cfg.visible_w = visible_w;
    cfg.visible_h = visible_h;
    cfg.vbi_lines = vbi_lines;
    cfg.hbi_width = hbi_width;
    cfg.canvas_w = hbi_width + visible_w;
    cfg.canvas_h = vbi_lines + visible_h;
    cfg.total_pixels = cfg.canvas_w * cfg.canvas_h;
    cfg.name = "custom";
    return cfg;
}

/* ===================================================================
 * Hamilton Encoder: RGB → Quaternion Latent
 * =================================================================== */

WubuLatent wubu_hamilton_encode(WubuRNG* rng, const float* images_rgb,
                                 int B, int H, int W) {
    WubuLatent latent;
    latent.B = B;
    latent.H = H;
    latent.W = W;

    int total_pixels = B * H * W;
    latent.quaternions = (float*)calloc((size_t)(total_pixels * 4), sizeof(float));
    latent.amplitude = (float*)calloc((size_t)total_pixels, sizeof(float));
    latent.context = (float*)calloc((size_t)(B * 3), sizeof(float));

    (void)rng;

    for (int b = 0; b < B; b++) {
        const float* img = images_rgb + b * H * W * 3;

        /* Context: mean color per batch */
        float img_mean[3] = {0.0f, 0.0f, 0.0f};
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = (y * W + x) * 3;
                img_mean[0] += img[idx + 0];
                img_mean[1] += img[idx + 1];
                img_mean[2] += img[idx + 2];
            }
        }
        float inv = 1.0f / (float)(H * W);
        latent.context[b * 3 + 0] = img_mean[0] * inv;
        latent.context[b * 3 + 1] = img_mean[1] * inv;
        latent.context[b * 3 + 2] = img_mean[2] * inv;

        /* Per-pixel quaternion from position + color */
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int pixel_idx = (b * H * W) + (y * W + x);
                int img_idx = (y * W + x) * 3;
                /* Per-pixel quaternion encoding (lossless)
                 * Store RGB directly in quaternion imaginary parts.
                 * q = (r, g, b, 1) — no normalization needed.
                 * The w=1 component marks this as a valid pixel (w=0 would be pure rotation).
                 * This allows perfect recovery: r = q[0], g = q[1], b = q[2]. */
                float r = img[img_idx + 0];
                float g = img[img_idx + 1];
                float b_col = img[img_idx + 2];

                latent.quaternions[pixel_idx * 4 + 0] = r;
                latent.quaternions[pixel_idx * 4 + 1] = g;
                latent.quaternions[pixel_idx * 4 + 2] = b_col;
                latent.quaternions[pixel_idx * 4 + 3] = 1.0f;

                /* Amplitude: luminance for weighting */
                latent.amplitude[pixel_idx] = 0.2989f * r + 0.5870f * g + 0.1140f * b_col;
            }
        }
    }

    return latent;
}

/* ===================================================================
 * Hamilton Decoder: Latent → RGB (coordinate sampling)
 * =================================================================== */

float* wubu_hamilton_decode(WubuRNG* rng, const WubuLatent* latent,
                              const float* coords, int N) {
    (void)rng;
    float* output = (float*)malloc((size_t)(N * 3) * sizeof(float));
    if (!output) return NULL;

    int H = latent->H;
    int W = latent->W;

    for (int i = 0; i < N; i++) {
        /* Coordinate in [-1,1] → pixel space */
        float cx = coords[i * 2 + 0];
        float cy = coords[i * 2 + 1];
        float xf = (cx + 1.0f) / 2.0f * (float)(W - 1);
        float yf = (cy + 1.0f) / 2.0f * (float)(H - 1);

        /* Bilinear sample from first batch entry */
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

        /* Sample quaternion + amplitude, reconstruct RGB */
        int base00 = y0 * W + x0;
        int base01 = y0 * W + x1;
        int base10 = y1 * W + x0;
        int base11 = y1 * W + x1;

        float q[4] = {0}, amp = 0;
        for (int c = 0; c < 4; c++) {
            q[c] = w00 * latent->quaternions[base00 * 4 + c] +
                   w01 * latent->quaternions[base01 * 4 + c] +
                   w10 * latent->quaternions[base10 * 4 + c] +
                   w11 * latent->quaternions[base11 * 4 + c];
        }
        amp = w00 * latent->amplitude[base00] +
              w01 * latent->amplitude[base01] +
              w10 * latent->amplitude[base10] +
              w11 * latent->amplitude[base11];

        /* Reconstruct RGB from quaternion (lossless)
         * Encoding: q = (r, g, b, 1)
         * Recovery: r = q[0], g = q[1], b = q[2] */
        float r = q[0];
        float g = q[1];
        float b = q[2];

        output[i * 3 + 0] = fminf(1.0f, fmaxf(0.0f, r));
        output[i * 3 + 1] = fminf(1.0f, fmaxf(0.0f, g));
        output[i * 3 + 2] = fminf(1.0f, fmaxf(0.0f, b));
    }

    return output;
}

void wubu_latent_free(WubuLatent* latent) {
    free(latent->quaternions);
    free(latent->amplitude);
    free(latent->context);
    latent->quaternions = NULL;
    latent->amplitude = NULL;
    latent->context = NULL;
}

/* ===================================================================
 * Audio Pipeline — VHF HBI Strip
 * =================================================================== */

float* wubu_audio_make_hbi_strip(const float* audio, int num_samples,
                                  const WubuCanvasConfig* config) {
    int target_size = config->canvas_h * config->hbi_width;
    float* strip = (float*)malloc((size_t)(target_size * 3) * sizeof(float));
    if (!strip) return NULL;

    for (int i = 0; i < target_size; i++) {
        float sample = 0.0f;
        if (i < num_samples) {
            sample = audio[i];
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
        }
        strip[i * 3 + 0] = sample;
        strip[i * 3 + 1] = sample;
        strip[i * 3 + 2] = sample;
    }

    return strip;
}

float* wubu_audio_decode_hbi_strip(const float* hbi_strip, int num_samples,
                                    const WubuCanvasConfig* config) {
    int target_size = config->canvas_h * config->hbi_width;
    float* audio = (float*)malloc((size_t)num_samples * sizeof(float));
    if (!audio) return NULL;

    for (int i = 0; i < num_samples; i++) {
        if (i < target_size) {
            /* Average the 3 channels back to mono */
            audio[i] = (hbi_strip[i * 3 + 0] + hbi_strip[i * 3 + 1] + hbi_strip[i * 3 + 2]) / 3.0f;
        } else {
            audio[i] = 0.0f;
        }
    }

    return audio;
}

/* ===================================================================
 * Canvas Compositor
 * =================================================================== */

float* wubu_compose_canvas_res(const float* vbi_block, const float* audio_hbi,
                                const float* visible, const WubuCanvasConfig* config) {
    int cw = config->canvas_w;
    int ch = config->canvas_h;
    int vw = config->visible_w;
    int vh = config->visible_h;
    int vbi = config->vbi_lines;
    int hbi = config->hbi_width;

    float* canvas = (float*)calloc((size_t)(cw * ch * 3), sizeof(float));
    if (!canvas) return NULL;

    /* 1. VBI block: top rows, full canvas width */
    for (int y = 0; y < vbi; y++) {
        for (int x = 0; x < cw; x++) {
            int src_idx = (y * cw + x) * 3;
            int dst_idx = (y * cw + x) * 3;
            if (vbi_block) {
                canvas[dst_idx + 0] = vbi_block[src_idx + 0];
                canvas[dst_idx + 1] = vbi_block[src_idx + 1];
                canvas[dst_idx + 2] = vbi_block[src_idx + 2];
            }
        }
    }

    /* 2. Audio HBI: left columns, visible height */
    for (int y = 0; y < vh; y++) {
        for (int x = 0; x < hbi; x++) {
            int src_idx = (y * hbi + x) * 3;
            int dst_idx = ((vbi + y) * cw + x) * 3;
            if (audio_hbi) {
                canvas[dst_idx + 0] = audio_hbi[src_idx + 0];
                canvas[dst_idx + 1] = audio_hbi[src_idx + 1];
                canvas[dst_idx + 2] = audio_hbi[src_idx + 2];
            }
        }
    }

    /* 3. Visible frame: right of HBI, below VBI */
    for (int y = 0; y < vh; y++) {
        for (int x = 0; x < vw; x++) {
            int src_idx = (y * vw + x) * 3;
            int dst_idx = ((vbi + y) * cw + hbi + x) * 3;
            if (visible) {
                canvas[dst_idx + 0] = visible[src_idx + 0];
                canvas[dst_idx + 1] = visible[src_idx + 1];
                canvas[dst_idx + 2] = visible[src_idx + 2];
            }
        }
    }

    return canvas;
}

float* wubu_extract_vbi(const float* canvas, const WubuCanvasConfig* config) {
    int cw = config->canvas_w;
    int vbi = config->vbi_lines;
    float* vbi_block = (float*)malloc((size_t)(vbi * cw * 3) * sizeof(float));
    if (!vbi_block) return NULL;

    for (int y = 0; y < vbi; y++) {
        memcpy(&vbi_block[y * cw * 3], &canvas[y * cw * 3], cw * 3 * sizeof(float));
    }
    return vbi_block;
}

float* wubu_extract_hbi(const float* canvas, const WubuCanvasConfig* config) {
    int cw = config->canvas_w;
    int vbi = config->vbi_lines;
    int hbi = config->hbi_width;
    int vh = config->visible_h;
    float* hbi_strip = (float*)malloc((size_t)(vh * hbi * 3) * sizeof(float));
    if (!hbi_strip) return NULL;

    for (int y = 0; y < vh; y++) {
        memcpy(&hbi_strip[y * hbi * 3], &canvas[((vbi + y) * cw) * 3], hbi * 3 * sizeof(float));
    }
    return hbi_strip;
}

float* wubu_extract_visible(const float* canvas, const WubuCanvasConfig* config) {
    int cw = config->canvas_w;
    int vbi = config->vbi_lines;
    int hbi = config->hbi_width;
    int vw = config->visible_w;
    int vh = config->visible_h;
    float* visible = (float*)malloc((size_t)(vh * vw * 3) * sizeof(float));
    if (!visible) return NULL;

    for (int y = 0; y < vh; y++) {
        memcpy(&visible[y * vw * 3], &canvas[((vbi + y) * cw + hbi) * 3], vw * 3 * sizeof(float));
    }
    return visible;
}

/* ===================================================================
 * Training Step
 * =================================================================== */

WubuTrainState wubu_train_step(const float* pred_rgb, const float* gt_rgb,
                                int B, int H, int W,
                                QController* qc) {
    WubuTrainState state = {0};
    int N = B * H * W;

    WubuLoss loss = wubu_compute_loss(pred_rgb, gt_rgb, N);
    state.composite_loss = loss.composite_loss;
    state.luma_loss = loss.luma_loss;
    state.phase_loss = loss.phase_loss;
    state.sat_loss = loss.sat_loss;

    QControllerConfig config = WUBU_Q_CONTROLLER_DEFAULT;
    wubu_q_controller_update(qc, loss.composite_loss, &config);

    state.q_status = qc->status_code;
    state.current_lr = qc->current_lr;
    state.step_count = qc->step_count;

    return state;
}
