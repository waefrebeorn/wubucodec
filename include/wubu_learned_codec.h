/*
 * wubu_learned_codec.h -- Learned Neural Codec for WuBu Compression (Inference)
 *
 * This is the C inference engine for the trained WuBu codec.
 * Training happens in Python (JAX/Flax), weights are exported to .bin files.
 *
 * Architecture (from symmetric_geometric_autoencoder.py):
 *   Encoder:
 *     1. PatchEmbed: Conv4x4 stride 4 → LayerNorm → [B, H*W, D]
 *     2. GlobalPool: mean over patches → [B, D]
 *     3. WuBuNestingEncoder: 4 levels [512, 256, 128, latent_dim]
 *        - Each level: flow MLP (context → GELU → residual)
 *        - Inter-level: learned rotation + mapping MLP
 *   Decoder:
 *     1. WuBuNestingDecoder: reverse 4 levels with skip connections
 *        - At each level: v += encoder_states[skip_idx]['v_global']
 *        - Same WuBuLevel + WuBuInterLevelTransition as encoder
 *        - Final: project + add to initial_patches (U-Net skip)
 *     2. ImagePatchDecoder: ResBlocks + upsample → Conv → RGB
 *
 * Usage:
 *   WubuLearnedCodec codec;
 *   WubuLearnedConfig cfg = wubu_learned_config_image(1024, 1024, 32, 8);
 *   wubu_learned_init(&codec, &cfg);
 *   wubu_learned_load_weights(&codec, "weights.bin");
 *
 *   WubuEncodeState state;
 *   float latent[32];
 *   wubu_learned_encode(&codec, image, latent, &state);
 *   wubu_learned_decode(&codec, latent, &state, output);
 */

#ifndef WUBU_LEARNED_CODEC_H
#define WUBU_LEARNED_CODEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_MAX_LEVELS 6
#define WUBU_MAX_DIM 512

typedef struct {
    int input_h, input_w;
    int patch_size;         /* default 4 */
    int embed_dim;          /* D (default 128) */
    int latent_dim;         /* final latent dimension */
    int num_levels;         /* default 4 */
    int level_dims[WUBU_MAX_LEVELS];
    int num_boundary_points[WUBU_MAX_LEVELS];
    int flow_mlp_hidden;
    int quant_bits;
} WubuLearnedConfig;

WubuLearnedConfig wubu_learned_config_image(int h, int w, int latent_dim, int quant_bits);

/* Encoder state — stores intermediate representations for decoder skip connections */
typedef struct {
    int num_levels;
    int level_dims[WUBU_MAX_LEVELS];
    /* Encoder states: v_global at each level (before transition) */
    float v_global[WUBU_MAX_LEVELS][WUBU_MAX_DIM];
    /* Initial patches for U-Net skip (Hp*Wp*D) */
    int Hp, Wp, D;
    float* patches;  /* [Hp*Wp*D], NULL if not stored */
} WubuEncodeState;

typedef struct {
    WubuLearnedConfig config;

    /* Patch embed */
    float* patch_weight;    /* [embed_dim, 3*patch_size*patch_size] */
    float* patch_bias, *patch_ln_gamma, *patch_ln_beta;

    /* Encoder levels */
    float* level_boundaries[WUBU_MAX_LEVELS];
    float* level_descriptor[WUBU_MAX_LEVELS];
    float* level_spread[WUBU_MAX_LEVELS];
    float* flow_w1[WUBU_MAX_LEVELS], *flow_b1[WUBU_MAX_LEVELS];
    float* flow_w2[WUBU_MAX_LEVELS], *flow_b2[WUBU_MAX_LEVELS];

    /* Encoder inter-level transitions */
    float* trans_rot[WUBU_MAX_LEVELS];
    float* trans_w1[WUBU_MAX_LEVELS], *trans_b1[WUBU_MAX_LEVELS];
    float* trans_w2[WUBU_MAX_LEVELS], *trans_b2[WUBU_MAX_LEVELS];

    /* Decoder levels (mirror of encoder — reverse direction) */
    float* dec_flow_w1[WUBU_MAX_LEVELS], *dec_flow_b1[WUBU_MAX_LEVELS];
    float* dec_flow_w2[WUBU_MAX_LEVELS], *dec_flow_b2[WUBU_MAX_LEVELS];

    /* Decoder inter-level transitions */
    float* dec_trans_rot[WUBU_MAX_LEVELS];
    float* dec_trans_w1[WUBU_MAX_LEVELS], *dec_trans_b1[WUBU_MAX_LEVELS];
    float* dec_trans_w2[WUBU_MAX_LEVELS], *dec_trans_b2[WUBU_MAX_LEVELS];

    /* Decoder final projection */
    float* dec_final_w, *dec_final_b;

    /* Image Patch Decoder weights */
    int res_channels[4];  /* Channel counts: [D, D, D/2, D/4] */
    /* ResBlock 0 (D ch) */
    float* res0_ln1_gamma, *res0_ln1_beta;
    float* res0_conv1_w, *res0_conv1_b;
    float* res0_ln2_gamma, *res0_ln2_beta;
    float* res0_conv2_w, *res0_conv2_b;
    /* ResBlock 1 (D/2 ch) */
    float* res1_ln1_gamma, *res1_ln1_beta;
    float* res1_conv1_w, *res1_conv1_b;
    float* res1_ln2_gamma, *res1_ln2_beta;
    float* res1_conv2_w, *res1_conv2_b;
    /* ResBlock 2 (D/4 ch) */
    float* res2_ln1_gamma, *res2_ln1_beta;
    float* res2_conv1_w, *res2_conv1_b;
    float* res2_ln2_gamma, *res2_ln2_beta;
    float* res2_conv2_w, *res2_conv2_b;
    /* Upsample convs */
    float* up1_w, *up1_b;  /* D → D/2 */
    float* up2_w, *up2_b;  /* D/2 → D/4 */
    /* Output */
    float* out_ln_gamma, *out_ln_beta;
    float* out_conv_w, *out_conv_b;  /* D/4 → 3 */

} WubuLearnedCodec;

int wubu_learned_init(WubuLearnedCodec* codec, WubuLearnedConfig* config);
void wubu_learned_free(WubuLearnedCodec* codec);

/* Load/save weights */
int wubu_learned_load_weights(WubuLearnedCodec* codec, const char* path);
int wubu_learned_save_weights(WubuLearnedCodec* codec, const char* path);

/* Encode: image → latent + state for decoder
 * Caller must provide latent buffer of size >= latent_dim + sum(level_dims)
 * The latent output is: [latent_dim bytes for latent] [encoder state data appended]
 * Actually, use WubuEncodeState to store intermediate results.
 */
void wubu_learned_encode(WubuLearnedCodec* codec, const float* image_hwc,
                           float* latent, WubuEncodeState* state);

/* Decode: latent + state → image (full decoder pipeline with skip connections) */
void wubu_learned_decode(WubuLearnedCodec* codec, const float* latent,
                           const WubuEncodeState* state, float* image_hwc);

/* Evaluate PSNR (encode + decode) */
float wubu_learned_eval_psnr(WubuLearnedCodec* codec, const float* image_hwc);

/* Compression ratio */
float wubu_learned_compression_ratio(WubuLearnedCodec* codec);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LEARNED_CODEC_H */
