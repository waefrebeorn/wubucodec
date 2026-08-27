/*
 * wubu_gaad_encoder.h -- GAAD + WuBu Nested Fractal Encoder
 *
 * Faithful C implementation of the WuBuGAADHybridGen_v0.2.py architecture.
 *
 * GAAD (Generalized Anti-Anisotropic Diffusion) uses:
 *   - Golden ratio (PHI) spiral for point cloud generation
 *   - Golden ratio recursive subdivision for multi-scale regions
 *   - Logarithmic spiral: r = a * exp(b * θ), b = log(PHI) / (π/2)
 *
 * WuBu nesting in Poincaré ball space:
 *   - Curvature: c = c_base * PHI^(level % 4 - 1.5)
 *   - Rotation: angle = softplus(param) * PHI^(level % 5 - 2) * π/4
 *   - Tangent flow MLP per level with residual connections
 *   - Inter-level transforms: logmap0 → rotate → MLP → expmap0
 */

#ifndef WUBU_GAAD_ENCODER_H
#define WUBU_GAAD_ENCODER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_GAAD_POINTS 8192
#define IMG_DIM 64
#define MAX_POINT_DIM 64
#define MAX_LEVELS 4
#define MAX_REGIONS 256

/* Golden ratio */
#define WUBU_PHI 1.6180339887498948482f
#define WUBU_LOG_PHI_OVER_PI_2 0.48121182505960347f

/* Poincaré ball geometry */
typedef struct {
    float c;           /* curvature */
    float c_inv;       /* 1/sqrt(c) */
    float one_minus_c; /* (1 - c) */
} PoincareBall;

/* Golden ratio spiral point */
typedef struct {
    float x, y;
    float scale;
} SpiralPoint;

/* Golden subdivision region */
typedef struct {
    float x1, y1, x2, y2;
} Rect;

/* WuBu nesting level */
typedef struct {
    int dim;
    float c;                    /* Curvature for this level */
    float c_base;
    int level_idx;
    int num_boundaries;
    int flow_hidden;
    int trans_hidden;
    int use_rotation;

    float* boundary_points;     /* [num_boundaries * dim] */
    float* descriptor;          /* [dim] */
    float spread;

    float* flow_w1;             /* [flow_hidden * comb_dim] */
    float* flow_b1;             /* [flow_hidden] */
    float* flow_w2;             /* [dim * flow_hidden] */
    float* flow_b2;             /* [dim] */

    float* rot_param;           /* rotation angle */
    float* trans_w1;            /* [trans_hidden * comb_dim] */
    float* trans_b1;            /* [trans_hidden] */
    float* trans_w2;            /* [next_dim * trans_hidden] */
    float* trans_b2;            /* [next_dim] */
} WubuLevel;

/* Full GAAD encoder */
typedef struct {
    SpiralPoint spiral_points[NUM_GAAD_POINTS];
    int num_spiral;

    Rect regions[MAX_REGIONS];
    int num_regions;

    WubuLevel levels[MAX_LEVELS];

    float proj_w[3 * MAX_POINT_DIM];
    float proj_b[MAX_POINT_DIM];

    float out_w[32 * MAX_POINT_DIM];  /* [latent_dim * point_dim] */
    float out_b[32];

    int latent_dim;
    int point_dim;
} WubuGaadEncoder;

/* Initialize encoder */
void wubu_gaad_init(WubuGaadEncoder* enc, int latent_dim);

/* Free encoder memory */
void wubu_gaad_free(WubuGaadEncoder* enc);

/* Encode image → latent */
void wubu_gaad_encode(WubuGaadEncoder* enc, const float* image, float* latent);

/* Encode with full point cloud (all 8192 points) */
void wubu_gaad_encode_full(WubuGaadEncoder* enc, const float* image,
                             float* latents, int* num_latents);

/* Poincaré ball operations */
PoincareBall poincare_ball(float c);
void poincare_project(float* x, int dim, PoincareBall* pb);
void poincare_expmap0(float* v, float* out, int dim, PoincareBall* pb);
void poincare_logmap0(float* x, float* out, int dim, PoincareBall* pb);
void poincare_mobius_add(float* x, float* y, float* out, int dim, PoincareBall* pb);

/* Golden ratio utilities */
int generate_phi_spiral(SpiralPoint* points, int num_points, int W, int H);
int golden_subdivide(Rect* rects, int W, int H, int num_target);

/* GELU activation */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

static inline float gelu_b(float x) {
    float c = 0.7978845608f;
    float inner = c * x * (1.0f + 0.044715f * x * x);
    float t = tanhf(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * x * x);
}

static inline float softplus(float x) {
    return logf(1.0f + expf(x));
}

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GAAD_ENCODER_H */
