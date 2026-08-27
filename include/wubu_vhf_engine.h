/*
 * wubu_vhf_engine.h -- Header for faithful VHF engine slermed from vhf_audio.py
 *
 * Components:
 *   - PositionalEncoding: sin/cos frequency encoding (10 freqs)
 *   - HamiltonEncoder: Multi-scale conv downsampling → quaternion+amplitude
 *   - VHFDecoder: [pos_enc, context, local_features] → 4-layer MLP → RGB
 *   - Full training step with HSL loss
 *   - Canvas compositing + audio strip generation
 */

#ifndef WUBU_VHF_ENGINE_H
#define WUBU_VHF_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canvas constants (from vhf_audio.py) */
#define VHF_CANVAS_W        656
#define VHF_CANVAS_H         525
#define VHF_VBI_LINES        45
#define VHF_VISIBLE_H        480
#define VHF_VISIBLE_W        640
#define VHF_AUDIO_HBI_WIDTH  16
#define VHF_TOTAL_PIXELS     (VHF_CANVAS_H * VHF_CANVAS_W)

#define VHF_POSENC_FREQS     10
#define VHF_DECODER_MLP_ITERS 4
#define VHF_DEFAULT_LATENT_GRID 96
#define VHF_DEFAULT_DMODEL   512
#define VHF_MAX_SCALE_LAYERS  6

/* ===================================================================
 * PositionalEncoding
 * =================================================================== */

typedef struct {
    int num_freqs;
    int input_dim;
    int output_dim;
} VHFPosEnc;

void vhf_posenc_init(VHFPosEnc* pe, int input_dim, int num_freqs);
void vhf_posenc_free(VHFPosEnc* pe);
void vhf_posenc_forward(const VHFPosEnc* pe, const float* coords, int N,
                         float* output);
/* Output: N × (input_dim * (1 + 2*num_freqs)) */

/* ===================================================================
 * HamiltonEncoder
 * Multi-scale Conv(4x4,stride2)→GELU downsampling + context aggregation
 * =================================================================== */

typedef struct {
    int latent_grid_size;
    int d_model;
    int num_scales;
    int features[VHF_MAX_SCALE_LAYERS];
    int scale_dims[VHF_MAX_SCALE_LAYERS];
    float* conv_down[VHF_MAX_SCALE_LAYERS];
    float* conv_down_bias[VHF_MAX_SCALE_LAYERS];
    float* conv_3x3_w;
    float* conv_3x3_b;
    float* conv_1x1_w;
    float* conv_1x1_b;
    int context_dim;
} VHFHamiltonEnc;

void vhf_hamilton_encoder_init(VHFHamiltonEnc* enc, int latent_grid_size, int d_model);
void vhf_hamilton_encoder_free(VHFHamiltonEnc* enc);

/* Encode a single image [H*W*3] → keys [LG*LG*5] + context [context_dim] */
void vhf_hamilton_encode(VHFHamiltonEnc* enc, const float* image_rgb,
                          int img_h, int img_w,
                          float* keys_out, float* context_out);

/* ===================================================================
 * VHFDecoder
 * [pos_enc, context, local_features] → 4-layer MLP → Dense(3) → tanh → RGB
 * =================================================================== */

typedef struct {
    int d_model;
    int context_dim;
    int local_feat_dim;
    int input_dim;
    int posenc_dim;
    float* mlp_w[VHF_DECODER_MLP_ITERS];
    float* mlp_b[VHF_DECODER_MLP_ITERS];
    int mlp_in[VHF_DECODER_MLP_ITERS];
    int mlp_out[VHF_DECODER_MLP_ITERS];
    float* out_w;
    float* out_b;
    VHFPosEnc posenc;
} VHFDec;

void vhf_decoder_init(VHFDec* dec, int d_model, int context_dim);
void vhf_decoder_free(VHFDec* dec);

/* Decode batch: keys [LH*LW*5] + context + coords [N,2] → output [N*3] */
void vhf_decode_batch(VHFDec* dec, const float* keys,
                        int LH, int LW,
                        const float* context,
                        const float* coords, int N,
                        float* output);

/* Decode single coordinate → rgb [3] */
void vhf_decode_per_sample(VHFDec* dec, const float* keys,
                             int LH, int LW,
                             const float* context,
                             float cx, float cy,
                             float* rgb_out);

/* ===================================================================
 * Loss computation
 * =================================================================== */

typedef struct {
    float composite_loss;
    float luma_loss;
    float phase_loss;
    float sat_loss;
} VHFLoss;

VHFLoss vhf_compute_loss(const float* pred_rgb, const float* gt_rgb, int N);

/* ===================================================================
 * Canvas compositing
 * Audio strip → HBI strip
 * =================================================================== */

float* vhf_generate_audio_strip(const float* audio, int num_samples);

/* Compose canvas from context + audio HBI + visible frame */
float* vhf_compose_canvas(const float* context, int context_dim,
                            const float* audio_hbi, const float* visible);

/* ===================================================================
 * Full training step (forward pass only - backprop in training module)
 * =================================================================== */

typedef struct {
    float composite_loss;
    float luma_loss;
    float phase_loss;
    float sat_loss;
    float current_lr;
    int q_status;
    int step_count;
} VHFTrainStepOutput;

VHFTrainStepOutput vhf_train_step_forward(
    VHFHamiltonEnc* enc,
    VHFDec* dec,
    const float* visible_frame,
    const float* audio_strip,
    const float* coords,
    const float* gt_rgb,
    int N,
    void* qc);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_VHF_ENGINE_H */
