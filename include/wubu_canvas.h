/*
 * wubu_canvas.h -- Multi-resolution WuBu Canvas System
 *
 * Configurable canvas for VHF video/audio compression pipeline.
 * Supports standard resolutions from 360P through 4K.
 *
 * Architecture:
 *   Hamilton Encoder: RGB frames → quaternion latent space (per-pixel)
 *   Hamilton Decoder: latent → RGB reconstruction (coordinate-sampled)
 *   Audio Pipeline: raw audio → HBI strip (frequency-encoded columns)
 *   Canvas Compositor: VBI + audio HBI + visible → full canvas
 *
 * Canvas layout:
 *   +-----------------------------------+
 *   |          VBI Block (top)          |  ← context padding
 *   +---+-------------------------------+
 *   | A |                               |
 *   | U |     Visible Video Frame       |
 *   | D |                               |
 *   | I |                               |
 *   | O |                               |
 *   +---+-------------------------------+
 *
 *   Total width  = AUDIO_HBI_WIDTH + VISIBLE_W
 *   Total height = VBI_LINES + VISIBLE_H
 *
 * Resolutions:
 *   360P:   640×360  → canvas 656×405
 *   480P:   854×480  → canvas 870×525
 *   720P:  1280×720  → canvas 1296×765
 *   1080P: 1920×1080 → canvas 1936×1125
 *   1440P: 2560×1440 → canvas 2576×1485
 *   4K:    3840×2160 → canvas 3856×2205
 */

#ifndef WUBU_CANVAS_H
#define WUBU_CANVAS_H

#include <stdint.h>
#include <stdbool.h>
#include "wubumath.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Standard Resolutions
 * =================================================================== */

typedef enum {
    WUBU_RES_360P  = 0,
    WUBU_RES_480P  = 1,
    WUBU_RES_720P  = 2,
    WUBU_RES_1080P = 3,
    WUBU_RES_1440P = 4,
    WUBU_RES_4K    = 5,
    WUBU_RES_COUNT
} WubuResolution;

typedef struct {
    int visible_w;        /* visible frame width */
    int visible_h;        /* visible frame height */
    int vbi_lines;        /* vertical blanking interval lines */
    int hbi_width;        /* horizontal blanking interval (audio columns) */
    int canvas_w;         /* total canvas width  = hbi_width + visible_w */
    int canvas_h;         /* total canvas height = vbi_lines + visible_h */
    int total_pixels;     /* canvas_w * canvas_h */
    const char* name;     /* human-readable name */
} WubuCanvasConfig;

/* Get config for a standard resolution */
const WubuCanvasConfig* wubu_canvas_get_config(WubuResolution res);

/* Get config by custom dimensions (for non-standard resolutions) */
WubuCanvasConfig wubu_canvas_make_config(int visible_w, int visible_h,
                                          int vbi_lines, int hbi_width);

/* ===================================================================
 * Latent Space — Hamilton Quaternion Encoder
 * =================================================================== */

/* Per-pixel latent: quaternion[4] + amplitude[1] + context[3] */
typedef struct {
    float* quaternions;   /* [B * H * W * 4] */
    float* amplitude;     /* [B * H * W * 1] */
    float* context;       /* [B * 3] — mean color per batch */
    int B, H, W;
} WubuLatent;

/* Encode RGB images into Hamilton latent space */
WubuLatent wubu_hamilton_encode(WubuRNG* rng, const float* images_rgb,
                                 int B, int H, int W);

/* Decode Hamilton latent back to RGB via coordinate sampling */
float* wubu_hamilton_decode(WubuRNG* rng, const WubuLatent* latent,
                              const float* coords, int N);

/* Free latent */
void wubu_latent_free(WubuLatent* latent);

/* ===================================================================
 * Audio Pipeline — VHF HBI Strip
 * =================================================================== */

/* Generate audio HBI strip from raw samples for given resolution */
float* wubu_audio_make_hbi_strip(const float* audio, int num_samples,
                                  const WubuCanvasConfig* config);

/* Decode audio from HBI strip back to raw samples */
float* wubu_audio_decode_hbi_strip(const float* hbi_strip, int num_samples,
                                    const WubuCanvasConfig* config);

/* ===================================================================
 * Canvas Compositor
 * =================================================================== */

/* Full compositor: VBI + HBI + visible → canvas (multi-resolution) */
float* wubu_compose_canvas_res(const float* vbi_block, const float* audio_hbi,
                                const float* visible, const WubuCanvasConfig* config);

/* Extract components from canvas */
float* wubu_extract_vbi(const float* canvas, const WubuCanvasConfig* config);
float* wubu_extract_hbi(const float* canvas, const WubuCanvasConfig* config);
float* wubu_extract_visible(const float* canvas, const WubuCanvasConfig* config);

/* ===================================================================
 * Training Loop — Multi-resolution
 * =================================================================== */

typedef struct {
    float composite_loss;
    float luma_loss;
    float phase_loss;
    float sat_loss;
    float current_lr;
    int q_status;
    int step_count;
    WubuResolution resolution;
} WubuTrainState;

/* Training step at given resolution */
WubuTrainState wubu_train_step(const float* pred_rgb, const float* gt_rgb,
                                int B, int H, int W,
                                QController* qc);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CANVAS_H */
