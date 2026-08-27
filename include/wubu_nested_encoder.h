/*
 * wubu_nested_encoder.h -- Multi-Level WuBu Nesting Encoder
 *
 * Implements the full WuBu Nesting architecture for image/video compression:
 *   - Multiple hyperbolic levels with learnable curvature/scale/spread
 *   - Tangent space rotation between levels via Hamilton product
 *   - Scale-aware exp/log maps for inter-level transitions
 *   - Boundary manifolds for sub-structure capture
 *   - Proper quantization respecting hyperbolic geometry
 *
 * This is the mathematically correct implementation of the WUBU compression
 * pipeline, faithful to the Python reference in wubu_nesting_impl.py.
 */

#ifndef WUBU_NESTED_ENCODER_H
#define WUBU_NESTED_ENCODER_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Configuration
 * =================================================================== */

#define WUBU_MAX_LEVELS 6
#define WUBU_MAX_DIM 256
#define WUBU_MAX_BOUNDARY_POINTS 16

typedef struct {
    int num_levels;                    /* number of nesting levels (2-6) */
    int input_dim;                     /* input dimension (e.g. 3 for RGB) */
    int hyperbolic_dims[WUBU_MAX_LEVELS]; /* dimension per level (decreasing) */
    int boundary_points[WUBU_MAX_LEVELS]; /* boundary points per level */
    float initial_curvatures[WUBU_MAX_LEVELS]; /* curvature c_i per level */
    float initial_scales[WUBU_MAX_LEVELS];     /* scale s_i per level */
    float initial_spreads[WUBU_MAX_LEVELS];    /* spread sigma_i per level */
    int quat_bits;                     /* bits per quaternion component (4-16) */
    int amp_bits;                      /* bits per amplitude value (4-16) */
    int use_tangent_flow;              /* enable tangent flow MLP */
    int use_level_descriptors;         /* enable level descriptors */
    int use_level_spread;              /* enable level spread */
} WubuNestedConfig;

/* Default configuration for image compression */
WubuNestedConfig wubu_nested_config_image(int width, int height, int base_bits);

/* ===================================================================
 * Model State
 * =================================================================== */

typedef struct {
    /* Configuration */
    WubuNestedConfig config;

    /* Learnable parameters */
    float log_curvature[WUBU_MAX_LEVELS];    /* log(c_i - min_c) */
    float log_scale[WUBU_MAX_LEVELS];        /* log(s_i - min_s) */
    float log_spread[WUBU_MAX_LEVELS];       /* log(sigma_i - min_sigma) */

    /* Input/output projection weights */
    float* input_proj;              /* [input_dim x hyperbolic_dims[0]] */
    float* output_proj;             /* [sum(dims) x output_dim] */

    /* Tangent combiner weights per level (small MLPs) */
    float* combiner_w1[WUBU_MAX_LEVELS]; /* [input_dim x hidden] */
    float* combiner_w2[WUBU_MAX_LEVELS]; /* [hidden x dim] */

    /* Rotation quaternions per inter-level transition */
    float rot_p[WUBU_MAX_LEVELS][4];   /* rotation quaternion p */
    float rot_q[WUBU_MAX_LEVELS][4];   /* rotation quaternion q */

    /* Boundary points */
    float* boundaries[WUBU_MAX_LEVELS]; /* [num_points x dim] */

    /* Level descriptors */
    float* level_descriptors[WUBU_MAX_LEVELS]; /* [dim] */

    /* Training state */
    int step_count;
    float learning_rate;

} WubuNestedEncoder;

/* ===================================================================
 * API
 * =================================================================== */

/* Initialize encoder with given config */
int wubu_nested_init(WubuNestedEncoder* enc, WubuNestedConfig* config);

/* Free all allocated memory */
void wubu_nested_free(WubuNestedEncoder* enc);

/* Encode: RGB image → compressed latent at all levels
 * Input: RGB float [H, W, 3] in [0,1]
 * Output: compressed latents at each level
 */
typedef struct {
    float* quaternions;   /* [N_total x 4] packed quaternion cloud */
    float* amplitudes;    /* [N_total] amplitude values */
    float* contexts;      /* [num_levels x 3] per-level context */
    int points_per_level[WUBU_MAX_LEVELS];
    int total_points;
    size_t raw_bytes;
    size_t comp_bytes;
} WubuCompressedImage;

WubuCompressedImage wubu_nested_encode(WubuNestedEncoder* enc,
                                        const float* rgb, int W, int H);

/* Decode: compressed latent → RGB image */
float* wubu_nested_decode(WubuNestedEncoder* enc,
                           const WubuCompressedImage* comp,
                           int W, int H);

/* Training step: encode → decode → compute loss → gradient update */
float wubu_nested_train_step(WubuNestedEncoder* enc,
                               const float* rgb, int W, int H);

/* Get current PSNR for debugging */
float wubu_nested_eval_psnr(WubuNestedEncoder* enc,
                               const float* rgb, int W, int H);

/* Free compressed image */
void wubu_compressed_free(WubuCompressedImage* comp);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_NESTED_ENCODER_H */
