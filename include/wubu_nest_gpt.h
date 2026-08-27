/*
 * wubu_nest_gpt.h -- WuBuNestGPT: Hyperbolic GPT with MLA attention
 *
 * Slermed from wubu_nest_gpt_numpy.py (bytropix/WUBUNEST_V2/)
 * Pure C11 implementation of Multi-head Latent Attention with
 * Poincare ball hyperbolic gyration position encoding.
 *
 * Features:
 *   - Multi-head Latent Attention (MLA): low-rank KV compression
 *   - Hyperbolic gyration position encoding (replaces RoPE)
 *   - SwiGLU FFN
 *   - LayerNorm + Residual connections
 *   - Autoregressive generation with top-k sampling
 */

#ifndef WUBU_NEST_GPT_H
#define WUBU_NEST_GPT_H

#include <stddef.h>
#include <stdbool.h>
#include "wubu_hyperbolic.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Model configuration
 * =================================================================== */

typedef struct {
    int vocab_size;       /* V */
    int d_model;          /* D - model dimension */
    int n_heads;          /* H - number of attention heads */
    int d_head;           /* d_h - per-head dimension (D / H) */
    int d_compressed;     /* d_c - compressed latent dim */
    int d_ff;             /* FFN hidden dimension */
    int n_layers;         /* N - number of transformer blocks */
    int dropout_rate;     /* dropout probability */
    int seed;             /* random seed */
    float init_scale;     /* weight initialization scale */
} WubuGPTConfig;

/* ===================================================================
 * Single attention block parameters
 * =================================================================== */

typedef struct {
    /* Layer norms */
    float* ln1_gamma;     /* [D] */
    float* ln1_beta;      /* [D] */
    float* ln2_gamma;     /* [D] */
    float* ln2_beta;      /* [D] */

    /* MLA projections */
    float* wq;            /* [D, H*d_h] */
    float* wdkv;          /* [D, d_c] */
    float* wuk;           /* [d_c, H*d_h] */
    float* wuv;           /* [d_c, H*d_h] */
    float* wqr;           /* [D, H*d_rope] */
    float* wkr;           /* [D, H*d_rope] */
    float* wo;            /* [H*d_h, D] */

    /* FFN */
    float* ffn1_w;        /* [D, d_ff] */
    float* ffn1_b;        /* [d_ff] */
    float* ffn2_w;        /* [d_ff, D] */
    float* ffn2_b;        /* [D] */
} WubuGPTBlock;

/* ===================================================================
 * Model state
 * =================================================================== */

typedef struct {
    WubuGPTConfig config;

    /* Token embeddings */
    float* wte;           /* [V, D] */
    float* lm_head_w;     /* [D, V] */

    /* Final layer norm */
    float* ln_f_gamma;    /* [D] */
    float* ln_f_beta;     /* [D] */

    /* Blocks */
    WubuGPTBlock* blocks; /* [N] */

    /* Dimensions cached */
    int D, H, d_h, d_c, d_rope, d_ff, N, V;
} WubuGPT;

/* ===================================================================
 * Initialize model (allocates all weights)
 * =================================================================== */

int wubu_gpt_init(WubuGPT* model, const WubuGPTConfig* config);

/* ===================================================================
 * Free model resources
 * =================================================================== */

void wubu_gpt_free(WubuGPT* model);

/* ===================================================================
 * Forward pass
 * =================================================================== */
float* wubu_gpt_forward(WubuGPT* model, const int* tokens, int B, int T, bool training);

/* ===================================================================
 * Compute cross-entropy loss
 * =================================================================== */

float wubu_gpt_compute_loss(WubuGPT* model, const int* targets, int B, int T);

/* ===================================================================
 * Generate tokens autoregressively
 * =================================================================== */

int* wubu_gpt_generate(WubuGPT* model, const int* prompt, int prompt_len,
                       int max_new_tokens, float temperature, int top_k,
                       int* out_len);

/* ===================================================================
 * Count total parameters
 * =================================================================== */

long wubu_gpt_count_params(const WubuGPT* model);

/* GAP-D012: embedding-input mode — language (or ANY modality already
 * embedded by manifold-CLIP heads) enters the transformer through its
 * continuous features instead of token-id lookup. feats: [B,T,d_model].
 * Returns hidden states [B,T,d_model]; caller pools/projects downstream. */
float* wubu_gpt_forward_embed(WubuGPT* model, const float* feats,
                              int B, int T, bool training);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_NEST_GPT_H */
