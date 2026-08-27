/*
 * wubu_learned_codec.c -- Inference engine for trained WuBu codec
 *
 * Faithful port of ALL 4 classes from symmetric_geometric_autoencoder.py:
 *   - ImagePatchEncoder  → PatchEmbed + LayerNorm
 *   - WuBuNestingEncoder → 4-level nesting + flow MLPs + transitions
 *   - WuBuNestingDecoder → reverse 4-level + skip connections + residual add
 *   - ImagePatchDecoder  → ResBlocks + upsample + Conv → RGB
 */

#include "wubu_learned_codec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===================================================================
 * Helper: GELU
 * =================================================================== */

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

/* ===================================================================
 * Helper: Linear forward — out = W @ x + b
 * W stored as [out_dim, in_dim] row-major
 * =================================================================== */

static void linear_fwd(const float* W, const float* b, const float* x,
                        float* out, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++)
            sum += W[o * in_dim + i] * x[i];
        out[o] = sum;
    }
}

/* ===================================================================
 * Helper: Conv2D 3x3 with SAME padding, stride 1
 * kernel: [out_ch, in_ch, 3, 3] flattened as [oc*ic*9 + ic*9 + ky*3 + kx]
 * =================================================================== */

static void conv2d_3x3_ch(
    const float* kernel, const float* bias,
    const float* input, float* output,
    int H, int W, int in_ch, int out_ch)
{
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = y + ky - 1;
                            int ix = x + kx - 1;
                            if (iy < 0) iy = 0;
                            if (iy >= H) iy = H - 1;
                            if (ix < 0) ix = 0;
                            if (ix >= W) ix = W - 1;
                            float in_val = input[(iy * W + ix) * in_ch + ic];
                            sum += in_val * kernel[oc * in_ch * 9 + ic * 9 + ky * 3 + kx];
                        }
                    }
                }
                output[(y * W + x) * out_ch + oc] = sum;
            }
        }
    }
}

/* ===================================================================
 * Helper: LayerNorm (over last dimension)
 * =================================================================== */

static void layernorm_fwd(const float* gamma, const float* beta,
                          const float* input, float* output,
                          int dim)
{
    float mean = 0;
    for (int i = 0; i < dim; i++) mean += input[i];
    mean /= (float)dim;
    float var = 0;
    for (int i = 0; i < dim; i++) {
        float d = input[i] - mean;
        var += d * d;
    }
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++)
        output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
}

/* ===================================================================
 * Helper: Nearest-neighbor upsample 2x
 * =================================================================== */

static void upsample2x_nearest(const float* input, float* output,
                                int H, int W, int C)
{
    for (int y = 0; y < H * 2; y++) {
        for (int x = 0; x < W * 2; x++) {
            int sy = y / 2;
            int sx = x / 2;
            for (int c = 0; c < C; c++) {
                output[(y * W * 2 + x) * C + c] = input[(sy * W + sx) * C + c];
            }
        }
    }
}

/* ===================================================================
 * Init / Free
 * =================================================================== */

WubuLearnedConfig wubu_learned_config_image(int h, int w, int latent_dim, int quant_bits) {
    WubuLearnedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_h = h;
    cfg.input_w = w;
    cfg.patch_size = 4;
    cfg.embed_dim = 128;
    cfg.latent_dim = latent_dim;
    cfg.num_levels = 4;
    cfg.level_dims[0] = 512;
    cfg.level_dims[1] = 256;
    cfg.level_dims[2] = 128;
    cfg.level_dims[3] = latent_dim;
    cfg.num_boundary_points[0] = 16;
    cfg.num_boundary_points[1] = 12;
    cfg.num_boundary_points[2] = 8;
    cfg.num_boundary_points[3] = 4;
    cfg.flow_mlp_hidden = 256;
    cfg.quant_bits = quant_bits;
    return cfg;
}

static float* alloc_zeros(int n) {
    return (float*)calloc((size_t)n, sizeof(float));
}

static float* alloc_ones(int n) {
    float* p = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) p[i] = 1.0f;
    return p;
}

int wubu_learned_init(WubuLearnedCodec* codec, WubuLearnedConfig* config) {
    memset(codec, 0, sizeof(WubuLearnedCodec));
    codec->config = *config;

    int D = config->embed_dim;
    int ps = config->patch_size;
    int patch_pix = 3 * ps * ps;

    /* Patch embed */
    codec->patch_weight = alloc_zeros(D * patch_pix);
    codec->patch_bias = alloc_zeros(D);
    codec->patch_ln_gamma = alloc_ones(D);
    codec->patch_ln_beta = alloc_zeros(D);

    /* Encoder levels */
    for (int l = 0; l < config->num_levels; l++) {
        int dim = config->level_dims[l];
        int nbp = config->num_boundary_points[l];
        int hidden = config->flow_mlp_hidden;
        int ctx_dim = 3 * dim + 1;

        codec->level_boundaries[l] = alloc_zeros(nbp * dim);
        codec->level_descriptor[l] = alloc_zeros(dim);
        codec->level_spread[l] = alloc_zeros(1);
        codec->flow_w1[l] = alloc_zeros(hidden * ctx_dim);
        codec->flow_b1[l] = alloc_zeros(hidden);
        codec->flow_w2[l] = alloc_zeros(dim * hidden);
        codec->flow_b2[l] = alloc_zeros(dim);

        if (l < config->num_levels - 1) {
            int next_dim = config->level_dims[l + 1];
            codec->trans_rot[l] = alloc_zeros(dim * dim);
            codec->trans_w1[l] = alloc_zeros(dim * 2 * dim);
            codec->trans_b1[l] = alloc_zeros(dim * 2);
            codec->trans_w2[l] = alloc_zeros(next_dim * dim * 2);
            codec->trans_b2[l] = alloc_zeros(next_dim);
        }
    }

    /* Decoder levels (reverse of encoder) */
    for (int l = 0; l < config->num_levels; l++) {
        int enc_l = config->num_levels - 1 - l;
        int dim = config->level_dims[enc_l];
        int hidden = config->flow_mlp_hidden;
        int ctx_dim = 3 * dim + 1;

        codec->dec_flow_w1[l] = alloc_zeros(hidden * ctx_dim);
        codec->dec_flow_b1[l] = alloc_zeros(hidden);
        codec->dec_flow_w2[l] = alloc_zeros(dim * hidden);
        codec->dec_flow_b2[l] = alloc_zeros(dim);

        if (l < config->num_levels - 1) {
            int next_enc_l = config->num_levels - 1 - (l + 1);
            int next_dim = config->level_dims[next_enc_l];
            codec->dec_trans_rot[l] = alloc_zeros(dim * dim);
            codec->dec_trans_w1[l] = alloc_zeros(dim * 2 * dim);
            codec->dec_trans_b1[l] = alloc_zeros(dim * 2);
            codec->dec_trans_w2[l] = alloc_zeros(next_dim * dim * 2);
            codec->dec_trans_b2[l] = alloc_zeros(next_dim);
        }
    }

    /* Decoder final projection */
    codec->dec_final_w = alloc_zeros(D * D);
    codec->dec_final_b = alloc_zeros(D);

    /* Image Patch Decoder weights */
    int ch0 = D;        /* 128 */
    int ch1 = ch0 / 2;  /* 64  */
    int ch2 = ch1 / 2;  /* 32  */
    codec->res_channels[0] = ch0;
    codec->res_channels[1] = ch0;
    codec->res_channels[2] = ch1;
    codec->res_channels[3] = ch2;

    /* ResBlock 0 (ch0 channels) */
    codec->res0_ln1_gamma = alloc_ones(ch0);
    codec->res0_ln1_beta = alloc_zeros(ch0);
    codec->res0_conv1_w = alloc_zeros(3*3*ch0*ch0);
    codec->res0_conv1_b = alloc_zeros(ch0);
    codec->res0_ln2_gamma = alloc_ones(ch0);
    codec->res0_ln2_beta = alloc_zeros(ch0);
    codec->res0_conv2_w = alloc_zeros(3*3*ch0*ch0);
    codec->res0_conv2_b = alloc_zeros(ch0);

    /* Upsample 1: ch0 → ch1 */
    codec->up1_w = alloc_zeros(3*3*ch0*ch1);
    codec->up1_b = alloc_zeros(ch1);

    /* ResBlock 1 (ch1 channels) */
    codec->res1_ln1_gamma = alloc_ones(ch1);
    codec->res1_ln1_beta = alloc_zeros(ch1);
    codec->res1_conv1_w = alloc_zeros(3*3*ch1*ch1);
    codec->res1_conv1_b = alloc_zeros(ch1);
    codec->res1_ln2_gamma = alloc_ones(ch1);
    codec->res1_ln2_beta = alloc_zeros(ch1);
    codec->res1_conv2_w = alloc_zeros(3*3*ch1*ch1);
    codec->res1_conv2_b = alloc_zeros(ch1);

    /* Upsample 2: ch1 → ch2 */
    codec->up2_w = alloc_zeros(3*3*ch1*ch2);
    codec->up2_b = alloc_zeros(ch2);

    /* ResBlock 2 (ch2 channels) */
    codec->res2_ln1_gamma = alloc_ones(ch2);
    codec->res2_ln1_beta = alloc_zeros(ch2);
    codec->res2_conv1_w = alloc_zeros(3*3*ch2*ch2);
    codec->res2_conv1_b = alloc_zeros(ch2);
    codec->res2_ln2_gamma = alloc_ones(ch2);
    codec->res2_ln2_beta = alloc_zeros(ch2);
    codec->res2_conv2_w = alloc_zeros(3*3*ch2*ch2);
    codec->res2_conv2_b = alloc_zeros(ch2);

    /* Output: LayerNorm + Conv3x3 → 3 */
    codec->out_ln_gamma = alloc_ones(ch2);
    codec->out_ln_beta = alloc_zeros(ch2);
    codec->out_conv_w = alloc_zeros(3*3*ch2*3);
    codec->out_conv_b = alloc_zeros(3);

    return 0;
}

void wubu_learned_free(WubuLearnedCodec* codec) {
    free(codec->patch_weight); free(codec->patch_bias);
    free(codec->patch_ln_gamma); free(codec->patch_ln_beta);
    free(codec->dec_final_w); free(codec->dec_final_b);

    for (int l = 0; l < WUBU_MAX_LEVELS; l++) {
        free(codec->level_boundaries[l]); free(codec->level_descriptor[l]);
        free(codec->level_spread[l]);
        free(codec->flow_w1[l]); free(codec->flow_b1[l]);
        free(codec->flow_w2[l]); free(codec->flow_b2[l]);
        free(codec->trans_rot[l]);
        free(codec->trans_w1[l]); free(codec->trans_b1[l]);
        free(codec->trans_w2[l]); free(codec->trans_b2[l]);
        free(codec->dec_flow_w1[l]); free(codec->dec_flow_b1[l]);
        free(codec->dec_flow_w2[l]); free(codec->dec_flow_b2[l]);
        free(codec->dec_trans_rot[l]);
        free(codec->dec_trans_w1[l]); free(codec->dec_trans_b1[l]);
        free(codec->dec_trans_w2[l]); free(codec->dec_trans_b2[l]);
    }

    free(codec->res0_ln1_gamma); free(codec->res0_ln1_beta);
    free(codec->res0_conv1_w); free(codec->res0_conv1_b);
    free(codec->res0_ln2_gamma); free(codec->res0_ln2_beta);
    free(codec->res0_conv2_w); free(codec->res0_conv2_b);
    free(codec->up1_w); free(codec->up1_b);

    free(codec->res1_ln1_gamma); free(codec->res1_ln1_beta);
    free(codec->res1_conv1_w); free(codec->res1_conv1_b);
    free(codec->res1_ln2_gamma); free(codec->res1_ln2_beta);
    free(codec->res1_conv2_w); free(codec->res1_conv2_b);
    free(codec->up2_w); free(codec->up2_b);

    free(codec->res2_ln1_gamma); free(codec->res2_ln1_beta);
    free(codec->res2_conv1_w); free(codec->res2_conv1_b);
    free(codec->res2_ln2_gamma); free(codec->res2_ln2_beta);
    free(codec->res2_conv2_w); free(codec->res2_conv2_b);

    free(codec->out_ln_gamma); free(codec->out_ln_beta);
    free(codec->out_conv_w); free(codec->out_conv_b);
}

/* ===================================================================
 * Weight I/O
 * =================================================================== */

int wubu_learned_load_weights(WubuLearnedCodec* codec, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    WubuLearnedConfig* cfg = &codec->config;
    int D = cfg->embed_dim, ps = cfg->patch_size;
    int patch_pix = 3 * ps * ps;

    fread(codec->patch_weight, sizeof(float), (size_t)(D * patch_pix), f);
    fread(codec->patch_bias, sizeof(float), (size_t)D, f);
    fread(codec->patch_ln_gamma, sizeof(float), (size_t)D, f);
    fread(codec->patch_ln_beta, sizeof(float), (size_t)D, f);

    /* Encoder */
    for (int l = 0; l < cfg->num_levels; l++) {
        int dim = cfg->level_dims[l];
        int nbp = cfg->num_boundary_points[l];
        int hidden = cfg->flow_mlp_hidden;
        int ctx_dim = 3 * dim + 1;

        fread(codec->level_boundaries[l], sizeof(float), (size_t)(nbp * dim), f);
        fread(codec->level_descriptor[l], sizeof(float), (size_t)dim, f);
        fread(codec->level_spread[l], sizeof(float), 1, f);
        fread(codec->flow_w1[l], sizeof(float), (size_t)(hidden * ctx_dim), f);
        fread(codec->flow_b1[l], sizeof(float), (size_t)hidden, f);
        fread(codec->flow_w2[l], sizeof(float), (size_t)(dim * hidden), f);
        fread(codec->flow_b2[l], sizeof(float), (size_t)dim, f);

        if (l < cfg->num_levels - 1) {
            int next_dim = cfg->level_dims[l + 1];
            fread(codec->trans_rot[l], sizeof(float), (size_t)(dim * dim), f);
            fread(codec->trans_w1[l], sizeof(float), (size_t)(dim * 2 * dim), f);
            fread(codec->trans_b1[l], sizeof(float), (size_t)(dim * 2), f);
            fread(codec->trans_w2[l], sizeof(float), (size_t)(next_dim * dim * 2), f);
            fread(codec->trans_b2[l], sizeof(float), (size_t)next_dim, f);
        }
    }

    /* Decoder levels (reverse order in file) */
    for (int l = 0; l < cfg->num_levels; l++) {
        int enc_l = cfg->num_levels - 1 - l;
        int dim = cfg->level_dims[enc_l];
        int hidden = cfg->flow_mlp_hidden;
        int ctx_dim = 3 * dim + 1;

        fread(codec->dec_flow_w1[l], sizeof(float), (size_t)(hidden * ctx_dim), f);
        fread(codec->dec_flow_b1[l], sizeof(float), (size_t)hidden, f);
        fread(codec->dec_flow_w2[l], sizeof(float), (size_t)(dim * hidden), f);
        fread(codec->dec_flow_b2[l], sizeof(float), (size_t)dim, f);

        if (l < cfg->num_levels - 1) {
            int next_enc_l = cfg->num_levels - 1 - (l + 1);
            int next_dim = cfg->level_dims[next_enc_l];
            fread(codec->dec_trans_rot[l], sizeof(float), (size_t)(dim * dim), f);
            fread(codec->dec_trans_w1[l], sizeof(float), (size_t)(dim * 2 * dim), f);
            fread(codec->dec_trans_b1[l], sizeof(float), (size_t)(dim * 2), f);
            fread(codec->dec_trans_w2[l], sizeof(float), (size_t)(next_dim * dim * 2), f);
            fread(codec->dec_trans_b2[l], sizeof(float), (size_t)next_dim, f);
        }
    }

    /* Decoder final projection */
    fread(codec->dec_final_w, sizeof(float), (size_t)(D * D), f);
    fread(codec->dec_final_b, sizeof(float), (size_t)D, f);

    /* Image Patch Decoder */
    int ch0 = D, ch1 = ch0 / 2, ch2 = ch1 / 2;

    fread(codec->res0_ln1_gamma, sizeof(float), (size_t)ch0, f);
    fread(codec->res0_ln1_beta, sizeof(float), (size_t)ch0, f);
    fread(codec->res0_conv1_w, sizeof(float), (size_t)(3*3*ch0*ch0), f);
    fread(codec->res0_conv1_b, sizeof(float), (size_t)ch0, f);
    fread(codec->res0_ln2_gamma, sizeof(float), (size_t)ch0, f);
    fread(codec->res0_ln2_beta, sizeof(float), (size_t)ch0, f);
    fread(codec->res0_conv2_w, sizeof(float), (size_t)(3*3*ch0*ch0), f);
    fread(codec->res0_conv2_b, sizeof(float), (size_t)ch0, f);

    fread(codec->up1_w, sizeof(float), (size_t)(3*3*ch0*ch1), f);
    fread(codec->up1_b, sizeof(float), (size_t)ch1, f);

    fread(codec->res1_ln1_gamma, sizeof(float), (size_t)ch1, f);
    fread(codec->res1_ln1_beta, sizeof(float), (size_t)ch1, f);
    fread(codec->res1_conv1_w, sizeof(float), (size_t)(3*3*ch1*ch1), f);
    fread(codec->res1_conv1_b, sizeof(float), (size_t)ch1, f);
    fread(codec->res1_ln2_gamma, sizeof(float), (size_t)ch1, f);
    fread(codec->res1_ln2_beta, sizeof(float), (size_t)ch1, f);
    fread(codec->res1_conv2_w, sizeof(float), (size_t)(3*3*ch1*ch1), f);
    fread(codec->res1_conv2_b, sizeof(float), (size_t)ch1, f);

    fread(codec->up2_w, sizeof(float), (size_t)(3*3*ch1*ch2), f);
    fread(codec->up2_b, sizeof(float), (size_t)ch2, f);

    fread(codec->res2_ln1_gamma, sizeof(float), (size_t)ch2, f);
    fread(codec->res2_ln1_beta, sizeof(float), (size_t)ch2, f);
    fread(codec->res2_conv1_w, sizeof(float), (size_t)(3*3*ch2*ch2), f);
    fread(codec->res2_conv1_b, sizeof(float), (size_t)ch2, f);
    fread(codec->res2_ln2_gamma, sizeof(float), (size_t)ch2, f);
    fread(codec->res2_ln2_beta, sizeof(float), (size_t)ch2, f);
    fread(codec->res2_conv2_w, sizeof(float), (size_t)(3*3*ch2*ch2), f);
    fread(codec->res2_conv2_b, sizeof(float), (size_t)ch2, f);

    fread(codec->out_ln_gamma, sizeof(float), (size_t)ch2, f);
    fread(codec->out_ln_beta, sizeof(float), (size_t)ch2, f);
    fread(codec->out_conv_w, sizeof(float), (size_t)(3*3*ch2*3), f);
    fread(codec->out_conv_b, sizeof(float), 3, f);

    fclose(f);
    return 0;
}

int wubu_learned_save_weights(WubuLearnedCodec* codec, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    WubuLearnedConfig* cfg = &codec->config;
    int D = cfg->embed_dim, ps = cfg->patch_size;
    int patch_pix = 3 * ps * ps;

    fwrite(codec->patch_weight, sizeof(float), (size_t)(D * patch_pix), f);
    fwrite(codec->patch_bias, sizeof(float), (size_t)D, f);
    fwrite(codec->patch_ln_gamma, sizeof(float), (size_t)D, f);
    fwrite(codec->patch_ln_beta, sizeof(float), (size_t)D, f);

    /* Encoder */
    for (int l = 0; l < cfg->num_levels; l++) {
        int dim = cfg->level_dims[l];
        int nbp = cfg->num_boundary_points[l];
        int hidden = cfg->flow_mlp_hidden;
        int ctx_dim = 3 * dim + 1;

        fwrite(codec->level_boundaries[l], sizeof(float), (size_t)(nbp * dim), f);
        fwrite(codec->level_descriptor[l], sizeof(float), (size_t)dim, f);
        fwrite(codec->level_spread[l], sizeof(float), 1, f);
        fwrite(codec->flow_w1[l], sizeof(float), (size_t)(hidden * ctx_dim), f);
        fwrite(codec->flow_b1[l], sizeof(float), (size_t)hidden, f);
        fwrite(codec->flow_w2[l], sizeof(float), (size_t)(dim * hidden), f);
        fwrite(codec->flow_b2[l], sizeof(float), (size_t)dim, f);

        if (l < cfg->num_levels - 1) {
            int next_dim = cfg->level_dims[l + 1];
            fwrite(codec->trans_rot[l], sizeof(float), (size_t)(dim * dim), f);
            fwrite(codec->trans_w1[l], sizeof(float), (size_t)(dim * 2 * dim), f);
            fwrite(codec->trans_b1[l], sizeof(float), (size_t)(dim * 2), f);
            fwrite(codec->trans_w2[l], sizeof(float), (size_t)(next_dim * dim * 2), f);
            fwrite(codec->trans_b2[l], sizeof(float), (size_t)next_dim, f);
        }
    }

    /* Decoder levels (reverse order) */
    for (int l = 0; l < cfg->num_levels; l++) {
        int enc_l = cfg->num_levels - 1 - l;
        int dim = cfg->level_dims[enc_l];
        int hidden = cfg->flow_mlp_hidden;
        int ctx_dim = 3 * dim + 1;

        fwrite(codec->dec_flow_w1[l], sizeof(float), (size_t)(hidden * ctx_dim), f);
        fwrite(codec->dec_flow_b1[l], sizeof(float), (size_t)hidden, f);
        fwrite(codec->dec_flow_w2[l], sizeof(float), (size_t)(dim * hidden), f);
        fwrite(codec->dec_flow_b2[l], sizeof(float), (size_t)dim, f);

        if (l < cfg->num_levels - 1) {
            int next_enc_l = cfg->num_levels - 1 - (l + 1);
            int next_dim = cfg->level_dims[next_enc_l];
            fwrite(codec->dec_trans_rot[l], sizeof(float), (size_t)(dim * dim), f);
            fwrite(codec->dec_trans_w1[l], sizeof(float), (size_t)(dim * 2 * dim), f);
            fwrite(codec->dec_trans_b1[l], sizeof(float), (size_t)(dim * 2), f);
            fwrite(codec->dec_trans_w2[l], sizeof(float), (size_t)(next_dim * dim * 2), f);
            fwrite(codec->dec_trans_b2[l], sizeof(float), (size_t)next_dim, f);
        }
    }

    fwrite(codec->dec_final_w, sizeof(float), (size_t)(D * D), f);
    fwrite(codec->dec_final_b, sizeof(float), (size_t)D, f);

    /* Image Patch Decoder */
    int ch0 = D, ch1 = ch0 / 2, ch2 = ch1 / 2;

    fwrite(codec->res0_ln1_gamma, sizeof(float), (size_t)ch0, f);
    fwrite(codec->res0_ln1_beta, sizeof(float), (size_t)ch0, f);
    fwrite(codec->res0_conv1_w, sizeof(float), (size_t)(3*3*ch0*ch0), f);
    fwrite(codec->res0_conv1_b, sizeof(float), (size_t)ch0, f);
    fwrite(codec->res0_ln2_gamma, sizeof(float), (size_t)ch0, f);
    fwrite(codec->res0_ln2_beta, sizeof(float), (size_t)ch0, f);
    fwrite(codec->res0_conv2_w, sizeof(float), (size_t)(3*3*ch0*ch0), f);
    fwrite(codec->res0_conv2_b, sizeof(float), (size_t)ch0, f);

    fwrite(codec->up1_w, sizeof(float), (size_t)(3*3*ch0*ch1), f);
    fwrite(codec->up1_b, sizeof(float), (size_t)ch1, f);

    fwrite(codec->res1_ln1_gamma, sizeof(float), (size_t)ch1, f);
    fwrite(codec->res1_ln1_beta, sizeof(float), (size_t)ch1, f);
    fwrite(codec->res1_conv1_w, sizeof(float), (size_t)(3*3*ch1*ch1), f);
    fwrite(codec->res1_conv1_b, sizeof(float), (size_t)ch1, f);
    fwrite(codec->res1_ln2_gamma, sizeof(float), (size_t)ch1, f);
    fwrite(codec->res1_ln2_beta, sizeof(float), (size_t)ch1, f);
    fwrite(codec->res1_conv2_w, sizeof(float), (size_t)(3*3*ch1*ch1), f);
    fwrite(codec->res1_conv2_b, sizeof(float), (size_t)ch1, f);

    fwrite(codec->up2_w, sizeof(float), (size_t)(3*3*ch1*ch2), f);
    fwrite(codec->up2_b, sizeof(float), (size_t)ch2, f);

    fwrite(codec->res2_ln1_gamma, sizeof(float), (size_t)ch2, f);
    fwrite(codec->res2_ln1_beta, sizeof(float), (size_t)ch2, f);
    fwrite(codec->res2_conv1_w, sizeof(float), (size_t)(3*3*ch2*ch2), f);
    fwrite(codec->res2_conv1_b, sizeof(float), (size_t)ch2, f);
    fwrite(codec->res2_ln2_gamma, sizeof(float), (size_t)ch2, f);
    fwrite(codec->res2_ln2_beta, sizeof(float), (size_t)ch2, f);
    fwrite(codec->res2_conv2_w, sizeof(float), (size_t)(3*3*ch2*ch2), f);
    fwrite(codec->res2_conv2_b, sizeof(float), (size_t)ch2, f);

    fwrite(codec->out_ln_gamma, sizeof(float), (size_t)ch2, f);
    fwrite(codec->out_ln_beta, sizeof(float), (size_t)ch2, f);
    fwrite(codec->out_conv_w, sizeof(float), (size_t)(3*3*ch2*3), f);
    fwrite(codec->out_conv_b, sizeof(float), 3, f);

    fclose(f);
    return 0;
}

/* ===================================================================
 * ENCODER: PatchEmbed → GlobalPool → WuBuNestingEncoder
 * =================================================================== */

void wubu_learned_encode(WubuLearnedCodec* codec, const float* image_hwc,
                           float* latent, WubuEncodeState* state) {
    WubuLearnedConfig* cfg = &codec->config;
    int H = cfg->input_h, W = cfg->input_w, ps = cfg->patch_size;
    int D = cfg->embed_dim;
    int Hp = H / ps, Wp = W / ps;
    int Np = Hp * Wp;
    int patch_pix = 3 * ps * ps;

    /* Step 1: Patch embedding */
    float* patches = (float*)calloc((size_t)(Np * D), sizeof(float));

    for (int py = 0; py < Hp; py++) {
        for (int px = 0; px < Wp; px++) {
            float patch_data[48];
            int idx = 0;
            for (int dy = 0; dy < ps; dy++) {
                for (int dx = 0; dx < ps; dx++) {
                    int iy = py * ps + dy, ix = px * ps + dx;
                    int img_idx = (iy * W + ix) * 3;
                    patch_data[idx++] = image_hwc[img_idx];
                    patch_data[idx++] = image_hwc[img_idx + 1];
                    patch_data[idx++] = image_hwc[img_idx + 2];
                }
            }
            float* out = patches + (py * Wp + px) * D;
            linear_fwd(codec->patch_weight, codec->patch_bias, patch_data, out,
                       patch_pix, D);
            layernorm_fwd(codec->patch_ln_gamma, codec->patch_ln_beta,
                         out, out, D);
        }
    }

    /* Store patches in state for U-Net skip */
    state->Hp = Hp;
    state->Wp = Wp;
    state->D = D;
    state->patches = (float*)malloc((size_t)(Np * D) * sizeof(float));
    memcpy(state->patches, patches, (size_t)(Np * D) * sizeof(float));

    /* Step 2: Global pool */
    float v_global[WUBU_MAX_DIM];
    memset(v_global, 0, (size_t)D * sizeof(float));
    for (int n = 0; n < Np; n++)
        for (int d = 0; d < D; d++)
            v_global[d] += patches[n * D + d];
    for (int d = 0; d < D; d++) v_global[d] /= (float)Np;

    free(patches);

    /* Step 3: WuBu Nesting Encoder */
    state->num_levels = cfg->num_levels;
    float* v = v_global;
    int v_dim = D;

    for (int l = 0; l < cfg->num_levels; l++) {
        int dim = cfg->level_dims[l];
        int nbp = cfg->num_boundary_points[l];
        int hidden = cfg->flow_mlp_hidden;

        state->level_dims[l] = dim;

        /* Project to level dimension */
        float v_level[WUBU_MAX_DIM];
        if (v_dim != dim) {
            for (int d = 0; d < dim; d++)
                v_level[d] = (d < v_dim) ? v[d] : 0.0f;
        } else {
            memcpy(v_level, v, (size_t)dim * sizeof(float));
        }

        /* Context: [v, mean(boundaries), descriptor, spread] */
        float ctx[WUBU_MAX_DIM * 3 + 1];
        float bound_mean[WUBU_MAX_DIM];
        for (int d = 0; d < dim; d++) {
            bound_mean[d] = 0;
            for (int p = 0; p < nbp; p++)
                bound_mean[d] += codec->level_boundaries[l][p * dim + d];
            bound_mean[d] /= (float)nbp;
        }

        int ci = 0;
        for (int d = 0; d < dim; d++) ctx[ci++] = v_level[d];
        for (int d = 0; d < dim; d++) ctx[ci++] = bound_mean[d];
        for (int d = 0; d < dim; d++) ctx[ci++] = codec->level_descriptor[l][d];
        ctx[ci++] = codec->level_spread[l][0];

        /* Flow MLP */
        float hidden_buf[WUBU_MAX_DIM];
        linear_fwd(codec->flow_w1[l], codec->flow_b1[l], ctx, hidden_buf,
                   3 * dim + 1, hidden);
        for (int h = 0; h < hidden; h++) hidden_buf[h] = gelu(hidden_buf[h]);

        float flow_vec[WUBU_MAX_DIM];
        linear_fwd(codec->flow_w2[l], codec->flow_b2[l], hidden_buf, flow_vec,
                   hidden, dim);

        /* Residual */
        float v_out[WUBU_MAX_DIM];
        for (int d = 0; d < dim; d++) v_out[d] = v_level[d] + flow_vec[d];

        /* Store encoder state for decoder skip */
        memcpy(state->v_global[l], v_out, (size_t)dim * sizeof(float));

        /* Inter-level transition */
        if (l < cfg->num_levels - 1) {
            int next_dim = cfg->level_dims[l + 1];

            /* Rotation */
            float v_rot[WUBU_MAX_DIM];
            for (int i = 0; i < dim; i++) {
                v_rot[i] = 0;
                for (int j = 0; j < dim; j++)
                    v_rot[i] += codec->trans_rot[l][i * dim + j] * v_out[j];
            }

            /* Mapping MLP */
            float mapped[WUBU_MAX_DIM * 2];
            linear_fwd(codec->trans_w1[l], codec->trans_b1[l], v_rot, mapped,
                       dim, dim * 2);
            for (int h = 0; h < dim * 2; h++) mapped[h] = gelu(mapped[h]);

            float v_next[WUBU_MAX_DIM];
            linear_fwd(codec->trans_w2[l], codec->trans_b2[l], mapped, v_next,
                       dim * 2, next_dim);

            v = v_next;
            v_dim = next_dim;
        } else {
            /* Last level: quantize and output latent */
            int lat_dim = cfg->latent_dim;
            float l_min = -1.0f, l_max = 1.0f;
            float range = l_max - l_min;
            int levels = (1 << cfg->quant_bits) - 1;
            for (int d = 0; d < lat_dim && d < dim; d++) {
                float normalized = (v_out[d] - l_min) / range;
                if (normalized < 0.0f) normalized = 0.0f;
                if (normalized > 1.0f) normalized = 1.0f;
                int q = (int)(normalized * (float)levels + 0.5f);
                if (q < 0) q = 0;
                if (q > levels) q = levels;
                latent[d] = l_min + ((float)q / (float)levels) * range;
            }
        }
    }
}

/* ===================================================================
 * DECODER: WuBuNestingDecoder → ImagePatchDecoder
 * =================================================================== */

void wubu_learned_decode(WubuLearnedCodec* codec, const float* latent,
                           const WubuEncodeState* state, float* image_hwc) {
    WubuLearnedConfig* cfg = &codec->config;
    int H = cfg->input_h, W = cfg->input_w, ps = cfg->patch_size;
    int D = cfg->embed_dim;
    int Hp = H / ps, Wp = W / ps;
    int Np = Hp * Wp;

    /* Step 1: WuBuNestingDecoder — reverse 4 levels with skip connections */
    /* Start from last encoder state (latent level) */
    int L = cfg->num_levels;
    int start_dim = state->level_dims[L - 1];

    float v_global[WUBU_MAX_DIM];
    memcpy(v_global, state->v_global[L - 1], (size_t)start_dim * sizeof(float));
    int v_dim = start_dim;

    for (int l = 0; l < L; l++) {
        /* Decoder level l corresponds to encoder level (L-1-l) */
        int enc_l = L - 1 - l;
        int dim = state->level_dims[enc_l];

        /* Skip connection: add encoder state */
        for (int d = 0; d < dim && d < v_dim; d++)
            v_global[d] += state->v_global[enc_l][d];

        /* WuBuLevel at this level */
        int nbp = cfg->num_boundary_points[enc_l];
        int hidden = cfg->flow_mlp_hidden;

        /* Context */
        float ctx[WUBU_MAX_DIM * 3 + 1];
        float bound_mean[WUBU_MAX_DIM];
        for (int d = 0; d < dim; d++) {
            bound_mean[d] = 0;
            for (int p = 0; p < nbp; p++)
                bound_mean[d] += codec->level_boundaries[enc_l][p * dim + d];
            bound_mean[d] /= (float)nbp;
        }

        int ci = 0;
        for (int d = 0; d < dim && d < v_dim; d++) ctx[ci++] = v_global[d];
        for (int d = ci; d < dim; d++) ctx[ci++] = 0.0f;
        for (int d = 0; d < dim; d++) ctx[ci++] = bound_mean[d];
        for (int d = 0; d < dim; d++) ctx[ci++] = codec->level_descriptor[enc_l][d];
        ctx[ci++] = codec->level_spread[enc_l][0];

        /* Flow MLP */
        float hidden_buf[WUBU_MAX_DIM];
        linear_fwd(codec->dec_flow_w1[l], codec->dec_flow_b1[l], ctx, hidden_buf,
                   3 * dim + 1, hidden);
        for (int h = 0; h < hidden; h++) hidden_buf[h] = gelu(hidden_buf[h]);

        float flow_vec[WUBU_MAX_DIM];
        linear_fwd(codec->dec_flow_w2[l], codec->dec_flow_b2[l], hidden_buf, flow_vec,
                   hidden, dim);

        /* Residual */
        float v_out[WUBU_MAX_DIM];
        for (int d = 0; d < dim; d++) v_out[d] = v_global[d] + flow_vec[d];

        /* Inter-level transition (decoder direction) */
        if (l < L - 1) {
            int next_enc_l = L - 1 - (l + 1);
            int next_dim = state->level_dims[next_enc_l];

            /* Rotation */
            float v_rot[WUBU_MAX_DIM];
            for (int i = 0; i < dim; i++) {
                v_rot[i] = 0;
                for (int j = 0; j < dim; j++)
                    v_rot[i] += codec->dec_trans_rot[l][i * dim + j] * v_out[j];
            }

            /* Mapping MLP */
            float mapped[WUBU_MAX_DIM * 2];
            linear_fwd(codec->dec_trans_w1[l], codec->dec_trans_b1[l], v_rot, mapped,
                       dim, dim * 2);
            for (int h = 0; h < dim * 2; h++) mapped[h] = gelu(mapped[h]);

            float v_next[WUBU_MAX_DIM];
            linear_fwd(codec->dec_trans_w2[l], codec->dec_trans_b2[l], mapped, v_next,
                       dim * 2, next_dim);

            memcpy(v_global, v_next, (size_t)next_dim * sizeof(float));
            v_dim = next_dim;
        } else {
            /* Final level: project and add to initial patches (U-Net skip) */
            float projected[WUBU_MAX_DIM];
            linear_fwd(codec->dec_final_w, codec->dec_final_b, v_out, projected,
                       dim, D);

            /* Add to initial patches (U-Net skip connection) */
            float* patches = state->patches;
            float* recon_patches = (float*)calloc((Np * D), sizeof(float));
            for (int n = 0; n < Np; n++)
                for (int d = 0; d < D; d++)
                    recon_patches[n * D + d] = patches[n * D + d] + projected[d];

            /* Step 2: ImagePatchDecoder */
            /* Reshape to spatial: [Hp, Wp, D] */
            float* h = recon_patches;

            /* ResBlock 0: LayerNorm → Conv3x3 → GELU → LayerNorm → Conv3x3 → skip */
            float* h_ln = (float*)calloc((Np * D), sizeof(float));
            for (int i = 0; i < Np; i++)
                layernorm_fwd(codec->res0_ln1_gamma, codec->res0_ln1_beta,
                             h + i * D, h_ln + i * D, D);
            float* tmp = (float*)calloc((Np * D), sizeof(float));
            conv2d_3x3_ch(codec->res0_conv1_w, codec->res0_conv1_b,
                           h_ln, tmp, Hp, Wp, D, D);
            for (int i = 0; i < Np * D; i++) tmp[i] = gelu(tmp[i]);
            for (int i = 0; i < Np; i++)
                layernorm_fwd(codec->res0_ln2_gamma, codec->res0_ln2_beta,
                             tmp + i * D, h_ln + i * D, D);
            conv2d_3x3_ch(codec->res0_conv2_w, codec->res0_conv2_b,
                           h_ln, tmp, Hp, Wp, D, D);
            for (int i = 0; i < Np * D; i++) tmp[i] += h[i];
            free(h_ln);
            free(h);

            /* Upsample 1: Hp×Wp → 2Hp×2Wp, channels D → D/2 */
            int H1 = Hp * 2, W1 = Wp * 2, ch1 = D / 2;
            float* h_up = (float*)calloc((H1 * W1 * D), sizeof(float));
            upsample2x_nearest(tmp, h_up, Hp, Wp, D);
            free(tmp);
            tmp = (float*)calloc((H1 * W1 * ch1), sizeof(float));
            conv2d_3x3_ch(codec->up1_w, codec->up1_b,
                           h_up, tmp, H1, W1, D, ch1);
            free(h_up);

            /* ResBlock 1 */
            h = tmp;
            h_ln = (float*)calloc((H1 * W1 * ch1), sizeof(float));
            for (int i = 0; i < H1 * W1; i++)
                layernorm_fwd(codec->res1_ln1_gamma, codec->res1_ln1_beta,
                             h + i * ch1, h_ln + i * ch1, ch1);
            float* tmp1 = (float*)calloc((H1 * W1 * ch1), sizeof(float));
            conv2d_3x3_ch(codec->res1_conv1_w, codec->res1_conv1_b,
                           h_ln, tmp1, H1, W1, ch1, ch1);
            for (int i = 0; i < H1 * W1 * ch1; i++) tmp1[i] = gelu(tmp1[i]);
            for (int i = 0; i < H1 * W1; i++)
                layernorm_fwd(codec->res1_ln2_gamma, codec->res1_ln2_beta,
                             tmp1 + i * ch1, h_ln + i * ch1, ch1);
            conv2d_3x3_ch(codec->res1_conv2_w, codec->res1_conv2_b,
                           h_ln, tmp1, H1, W1, ch1, ch1);
            for (int i = 0; i < H1 * W1 * ch1; i++) tmp1[i] += h[i];
            free(h_ln);
            free(h);

            /* Upsample 2: 2Hp×2Wp → 4Hp×4Wp = H×W, channels D/2 → D/4 */
            int H2 = H1 * 2, W2 = W1 * 2, ch2 = ch1 / 2;
            h_up = (float*)calloc((H2 * W2 * ch1), sizeof(float));
            upsample2x_nearest(tmp1, h_up, H1, W1, ch1);
            free(tmp1);
            tmp = (float*)calloc((H2 * W2 * ch2), sizeof(float));
            conv2d_3x3_ch(codec->up2_w, codec->up2_b,
                           h_up, tmp, H2, W2, ch1, ch2);
            free(h_up);

            /* ResBlock 2 */
            h = tmp;
            h_ln = (float*)calloc((H2 * W2 * ch2), sizeof(float));
            for (int i = 0; i < H2 * W2; i++)
                layernorm_fwd(codec->res2_ln1_gamma, codec->res2_ln1_beta,
                             h + i * ch2, h_ln + i * ch2, ch2);
            float* tmp2 = (float*)calloc((H2 * W2 * ch2), sizeof(float));
            conv2d_3x3_ch(codec->res2_conv1_w, codec->res2_conv1_b,
                           h_ln, tmp2, H2, W2, ch2, ch2);
            for (int i = 0; i < H2 * W2 * ch2; i++) tmp2[i] = gelu(tmp2[i]);
            for (int i = 0; i < H2 * W2; i++)
                layernorm_fwd(codec->res2_ln2_gamma, codec->res2_ln2_beta,
                             tmp2 + i * ch2, h_ln + i * ch2, ch2);
            conv2d_3x3_ch(codec->res2_conv2_w, codec->res2_conv2_b,
                           h_ln, tmp2, H2, W2, ch2, ch2);
            for (int i = 0; i < H2 * W2 * ch2; i++) tmp2[i] += h[i];
            free(h_ln);
            free(h);

            /* Output: LayerNorm → Conv3x3 → tanh → RGB */
            h = tmp2;
            h_ln = (float*)calloc((H2 * W2 * ch2), sizeof(float));
            for (int i = 0; i < H2 * W2; i++)
                layernorm_fwd(codec->out_ln_gamma, codec->out_ln_beta,
                             h + i * ch2, h_ln + i * ch2, ch2);

            float* out_conv = (float*)calloc((H2 * W2 * 3), sizeof(float));
            conv2d_3x3_ch(codec->out_conv_w, codec->out_conv_b,
                           h_ln, out_conv, H2, W2, ch2, 3);

            /* tanh + clamp to [0,1] */
            for (int i = 0; i < H * W * 3; i++) {
                float val = tanhf(out_conv[i]);
                image_hwc[i] = fminf(1.0f, fmaxf(0.0f, val * 0.5f + 0.5f));
            }

            free(out_conv);
            free(h_ln);
            free(h);
        }
    }
}

/* ===================================================================
 * PSNR evaluation
 * =================================================================== */

float wubu_learned_eval_psnr(WubuLearnedCodec* codec, const float* image_hwc) {
    WubuLearnedConfig* cfg = &codec->config;
    int H = cfg->input_h, W = cfg->input_w;

    float* latent = (float*)calloc((size_t)cfg->latent_dim, sizeof(float));
    float* output = (float*)calloc((size_t)(H * W * 3), sizeof(float));

    WubuEncodeState state;
    memset(&state, 0, sizeof(state));
    wubu_learned_encode(codec, image_hwc, latent, &state);
    wubu_learned_decode(codec, latent, &state, output);

    float mse = 0;
    for (int i = 0; i < H * W * 3; i++) {
        float d = image_hwc[i] - output[i];
        mse += d * d;
    }
    mse /= (float)(H * W * 3);

    free(latent);
    free(output);
    if (state.patches) free(state.patches);

    return (mse < 1e-10f) ? 100.0f : 10.0f * log10f(1.0f / mse);
}

float wubu_learned_compression_ratio(WubuLearnedCodec* codec) {
    float raw = (float)(codec->config.input_h * codec->config.input_w * 3 * sizeof(float));
    float comp = (float)(codec->config.latent_dim * codec->config.quant_bits) / 8.0f;
    return raw / comp;
}
