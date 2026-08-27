/*
 * wubumath.h -- WuBuMath: Pure C11 mathematical & media encoding library
 *
 * Slermed from vhf_audio.py (bytropix/AUDIO/wubusynth/vhf_audio.py)
 * Faithful C11 translation of the Hamilton VHF encoder/decoder pipeline
 *
 * Features:
 *   - PRNG (SplitMix64, JAX-compatible)
 *   - RGB/HSL color manifold
 *   - Circular L1 Loss
 *   - Positional encoding (sin/cos frequency bands)
 *   - Hamilton Encoder (quaternion latent space)
 *   - VHF Decoder (coordinate-sampled reconstruction)
 *   - Q-Controller (adaptive learning rate)
 *   - Audio strip generation (FM HBI encoding)
 *   - Canvas compositing (VBI + HBI + visible)
 */

#ifndef WUBUMATH_H
#define WUBUMATH_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Version
 * =================================================================== */

#define WUBU_MATH_VERSION_MAJOR 0
#define WUBU_MATH_VERSION_MINOR 1
#define WUBU_MATH_VERSION_PATCH 0

/* ===================================================================
 * Canvas Constants (from vhf_audio.py)
 * =================================================================== */

#define VBI_LINES           45
#define VISIBLE_H           480
#define CANVAS_H            (VISIBLE_H + VBI_LINES)  /* 525 */
#define AUDIO_HBI_WIDTH     16
#define VISIBLE_W           640
#define CANVAS_W            (AUDIO_HBI_WIDTH + VISIBLE_W)  /* 656 */
#define TOTAL_PIXELS        (CANVAS_H * CANVAS_W)  /* 344520 */

/* ===================================================================
 * PRNG: SplitMix64 (JAX-compatible)
 * =================================================================== */

typedef struct {
    uint64_t state[2];
} WubuRNG;

static inline void wubu_rng_init(WubuRNG* rng, uint64_t seed) {
    rng->state[0] = seed;
    rng->state[1] = seed ^ 0x9e3779b97f4a7c15ULL;
}

static inline uint64_t wubu_rng_next(WubuRNG* rng) {
    uint64_t* s = rng->state;
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline float wubu_rng_uniform(WubuRNG* rng, float min_val, float max_val) {
    uint64_t z = wubu_rng_next(rng);
    float u = (float)(z >> 8) / (float)(1ULL << 56);
    return min_val + u * (max_val - min_val);
}

static inline int wubu_rng_randint(WubuRNG* rng, int min_val, int max_val) {
    uint64_t z = wubu_rng_next(rng);
    int range = max_val - min_val;
    if (range <= 0) return min_val;
    return min_val + (int)(z % (uint64_t)range);
}

static inline void wubu_rng_split(WubuRNG* rng, uint64_t* lo, uint64_t* hi) {
    *lo = wubu_rng_next(rng);
    *hi = wubu_rng_next(rng);
}

/* ===================================================================
 * Color Manifold: RGB <-> HSL (from vhf_audio.py)
 * =================================================================== */

typedef struct { float h, s, l; } WubuHSL;
typedef struct { float r, g, b; } WubuRGB;
typedef struct { float r, g, b, a; } WubuRGBA;

/* RGB [0,1] -> HSL [0,1] */
WubuHSL wubu_rgb_to_hsl(WubuRGB rgb);
WubuRGB wubu_hsl_to_rgb(WubuHSL hsl);

/* RGB [0,1] -> grayscale [0,1] */
float wubu_rgb_to_grayscale(WubuRGB rgb);

/* Circular L1 Loss (for hue manifold) */
float wubu_circular_l1_loss(float pred, float target);

/* ===================================================================
 * Positional Encoding (from vhf_audio.py PositionalEncoding)
 * =================================================================== */

/* Encode input features [D] with sin/cos frequency bands
 * Output: [D + 2 * num_freqs * D] (original + sin + cos for each freq)
 * arena: optional arena allocator (pass NULL for malloc)
 */
float* wubu_positional_encode(const float* x, int D, int num_freqs);

/* ===================================================================
 * Q-Controller (from vhf_audio.py)
 * =================================================================== */

typedef struct {
    float* q_table;           /* [num_lr_actions] */
    float* metric_history;    /* [metric_history_len] */
    float current_lr;
    float exploration_rate;
    int   step_count;
    int   last_action_idx;
    int   status_code;        /* 0=warmup, 1=improving, 2=stagnated */
} QController;

typedef struct {
    int   num_lr_actions;
    float lr_change_factors[5];  /* max 5 factors */
    float learning_rate_q;
    float lr_min;
    float lr_max;
    int   metric_history_len;
    float exploration_rate_q;
    float min_exploration_rate;
    float exploration_decay;
    int   warmup_steps;
    float warmup_lr_start;
} QControllerConfig;

void wubu_q_controller_init(QController* qc, const QControllerConfig* config);
void wubu_q_controller_choose_action(QController* qc, WubuRNG* rng, const QControllerConfig* config, float target_lr);
void wubu_q_controller_update(QController* qc, float metric_value, const QControllerConfig* config);
void wubu_q_controller_free(QController* qc);

/* Default config (matches vhf_audio.py defaults) */
extern const QControllerConfig WUBU_Q_CONTROLLER_DEFAULT;

/* ===================================================================
 * Hamilton Encoder (from vhf_audio.py HamiltonEncoder)
 * =================================================================== */

typedef struct {
    /* Simplified 5-parameter latent: quaternion[4] + amplitude[1] */
    int latent_grid_size;
    int d_model;
} HamiltonEncoder;

typedef struct {
    float* quaternions;   /* [B * H * W * 4] */
    float* amplitude;     /* [B * H * W * 1] */
    float* context;       /* [B * context_dim] */
    int B, H, W;
    int context_dim;
} HamiltonKeys;

void wubu_hamilton_encoder_init(HamiltonEncoder* enc, int latent_grid_size, int d_model);
void wubu_hamilton_encoder_free(HamiltonEncoder* enc);

/* Encode RGB images [-1,1] into Hamilton keys */
/* For the slerm: simplified version using procedural encoding (no trained weights) */
HamiltonKeys wubu_hamilton_encode_procedural(WubuRNG* rng, const float* images_rgb, int B, int img_size);

void wubu_hamilton_keys_free(HamiltonKeys* keys);

/* ===================================================================
 * VHF Decoder (from vhf_audio.py VHFDecoder)
 * =================================================================== */

typedef struct {
    int d_model;
} VHFDecoder;

void wubu_vhf_decoder_init(VHFDecoder* dec, int d_model);
void wubu_vhf_decoder_free(VHFDecoder* dec);

/* Decode Hamilton keys + context + coords -> RGB [B, N, 3] in [-1,1] */
/* Procedural version: uses coordinate-based sampling with positional encoding */
float* wubu_vhf_decode_procedural(WubuRNG* rng, const HamiltonKeys* keys,
                                   const float* coords, int N);

/* ===================================================================
 * Audio Strip Generation (from vhf_audio.py video_audio_generator)
 * =================================================================== */

/* Generate HBI audio strip from raw audio samples
 * audio_samples: raw audio in [-1,1], length = CANVAS_H * AUDIO_HBI_WIDTH
 * Output: [CANVAS_H * AUDIO_HBI_WIDTH * 3] RGB (grayscale replicated to 3 channels)
 */
float* wubu_generate_audio_strip(const float* audio_samples, int num_samples);

/* ===================================================================
 * Canvas Compositing (from vhf_audio.py eval_canvases)
 * =================================================================== */

/* Composite a full canvas from components:
 *   vbi_block: [VBI_LINES * CANVAS_W * 3] - context padding
 *   audio_hbi: [VISIBLE_H * AUDIO_HBI_WIDTH * 3] - audio strip visible portion
 *   visible:   [VISIBLE_H * VISIBLE_W * 3] - video frame
 * Output: [CANVAS_H * CANVAS_W * 3] - full canvas
 */
float* wubu_compose_canvas(const float* vbi_block, const float* audio_hbi, const float* visible);

/* ===================================================================
 * Loss Computation (from vhf_audio.py train_step)
 * =================================================================== */

typedef struct {
    float composite_loss;
    float luma_loss;
    float phase_loss;
    float sat_loss;
} WubuLoss;

/* Compute the full VHF loss manifold
 * pred_rgb: [N * 3] predicted RGB in [-1,1]
 * gt_rgb:   [N * 3] ground truth RGB in [-1,1]
 */
WubuLoss wubu_compute_loss(const float* pred_rgb, const float* gt_rgb, int N);

/* ===================================================================
 * Utility: Bilinear Sampling (from vhf_audio.py map_coordinates)
 * =================================================================== */

/* Sample image [H, W, C] at coordinates [N, 2] (x,y in pixel space) */
float* wubu_bilinear_sample(const float* image, int H, int W, int C,
                            const float* coords, int N);

/* ===================================================================
 * Utility: Box Blur 5x5 (separable)
 * =================================================================== */

float* wubu_box_blur_5x5(const float* image, int H, int W);

#ifdef __cplusplus
}
#endif

/* ===================================================================
 * VHF Audio Model (slermed from vhf_audio.py)
 *
 * The VHF Hamilton Modulator + Audio pipeline:
 *   - HamiltonEncoder: Quaternion latent space from RGB images
 *   - VHFDecoder: Coordinate-sampled reconstruction
 *   - VideoVHFTrainer: Full training loop with audio injection
 *   - Video/audio data pipeline with HBI audio strip encoding
 * =================================================================== */

/* Hamilton Encoder Config */
typedef struct {
    int latent_grid_size;  /* default 96 */
    int d_model;           /* default 512 */
} VHFEncoderConfig;

typedef struct {
    float* quaternions;   /* [B * H * W * 4] */
    float* amplitude;     /* [B * H * W * 1] */
    float* context;       /* [B * 3] */
    int B, H, W;
} VHFEncoded;

/* VHF Decoder Config */
typedef struct {
    int d_model;           /* default 512 */
} VHFDecoderConfig;

/* Audio Strip Config */
typedef struct {
    int sample_rate;       /* default 44100 */
    int canvas_w;          /* 656 */
    int canvas_h;          /* 525 */
    int vbi_lines;         /* 45 */
    int visible_h;         /* 480 */
    int audio_hbi_width;   /* 16 */
} VHAudioConfig;

/* Training State */
typedef struct {
    float composite_loss;
    float luma_loss;
    float phase_loss;
    float sat_loss;
    float current_lr;
    int q_status;
    int step_count;
} VHFTrainingState;

extern const VHFEncoderConfig VHF_ENCODER_DEFAULT;
extern const VHFDecoderConfig VHF_DECODER_DEFAULT;
extern const VHAudioConfig VHF_AUDIO_DEFAULT;

/* Hamilton Encoder: RGB images [-1,1] -> quaternion latent space */
VHFEncoded wubu_vhf_encode(WubuRNG* rng, const float* images_rgb, int B, int img_size);

/* VHF Decoder: latent + context + coords -> RGB [-1,1] */
float* wubu_vhf_decode(WubuRNG* rng, const VHFEncoded* encoded,
                        const float* coords, int N);

void wubu_vhf_encoded_free(VHFEncoded* encoded);

/* Generate audio HBI strip: raw samples [-1,1] -> [canvas_h * audio_hbi_width * 3] */
float* wubu_vhf_generate_audio_strip(const float* audio, int num_samples,
                                      const VHAudioConfig* config);

/* Compose full canvas: VBI + audio HBI + visible -> [canvas_h * canvas_w * 3] */
float* wubu_vhf_compose_canvas(const float* vbi_block, const float* audio_hbi,
                                const float* visible, const VHAudioConfig* config);

/* Training step: compute loss + update Q-controller */
VHFTrainingState wubu_vhf_train_step(const float* pred_rgb, const float* gt_rgb,
                                       int pixels_per_step, QController* qc);

/* ===================================================================
 * Quaternion Operations (slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * =================================================================== */

#include "wubu_quaternion.h"

/* ===================================================================
 * Hyperbolic Geometry: Poincare Ball (slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * =================================================================== */

#include "wubu_hyperbolic.h"

/* ===================================================================
 * Riemannian SGD Optimizer (slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * =================================================================== */

#include "wubu_riemannian_sgd.h"

/* ===================================================================
 * Parallel Transport on Poincare Ball (slermed from FullyHyperbolicWuBuNestingModel)
 * =================================================================== */

#include "wubu_parallel_transport.h"

/* ===================================================================
 * Tangent Flow (slermed from HyperbolicWuBuNestingLevel)
 * =================================================================== */

#include "wubu_tangent_flow.h"

/* ===================================================================
 * WuBuNestGPT: Hyperbolic GPT with MLA (slermed from wubu_nest_gpt_numpy.py)
 * =================================================================== */

#include "wubu_nest_gpt.h"

/* ===================================================================
 * Multi-Resolution Canvas System (slermed from VHF pipeline)
 * =================================================================== */

#include "wubu_canvas.h"

/* ===================================================================
 * Flow Matching on Poincaré Ball (new — temporal coherence)
 * =================================================================== */

#include "wubu_flow_matching.h"

/* ===================================================================
 * Latent Space Compression Codec
 * =================================================================== */

#include "wubu_latent_codec.h"

/* ===================================================================
 * Hamilton Quaternion Operations (new — proper WUBU math)
 * =================================================================== */

#include "wubu_quaternion_ops.h"

/* ===================================================================
 * Multi-Level Nested Encoder (new — proper WUBU nesting)
 * =================================================================== */

#include "wubu_nested_encoder.h"

/* ===================================================================
 * Learned Neural Codec (new — trained encoder/decoder)
 * =================================================================== */

#include "wubu_learned_codec.h"
#include "wubu_gaad_encoder.h"

#endif /* WUBUMATH_H */
