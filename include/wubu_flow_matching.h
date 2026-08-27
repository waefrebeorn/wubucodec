/*
 * wubu_flow_matching.h -- Flow Matching on Poincaré Ball for Hamilton Quaternion Point Clouds
 *
 * Slermed from first principles: flow matching for temporal coherence between
 * key-frame Hamilton latent spaces.
 *
 * Architecture:
 *   Key frames: RGB → Hamilton quaternion latent (Poincaré ball points)
 *   Motion: Flow matching learns velocity field v_t(x) on tangent space T_x M
 *   Interpolation: probability path μ_t = (1-t)·x_0 + t·x_1 on manifold (geodesic)
 *   Training: regress velocity field to match geodesic tangent vectors
 *   Inference: ODE solve from x_0 to generate intermediate frames
 *
 * Math:
 *   - x_0, x_1: quaternion latent points on Poincaré ball (from key frames)
 *   - Geodesic interpolation: μ_t = exp_{x_0}(t · log_{x_0}(x_1))
 *   - Target velocity: d/dt μ_t = parallel transport of log_{x_0}(x_1) along geodesic
 *   - Learned velocity: v_θ(t, x) — MLP on tangent space at x
 *   - Loss: E[||v_θ(t, μ_t) - d/dt μ_t||²]
 *   - Inference: dx/dt = v_θ(t, x), x(0) ~ N(0,σ²I), solve ODE to x(1)
 */

#ifndef WUBU_FLOW_MATCHING_H
#define WUBU_FLOW_MATCHING_H

#include "wubu_hyperbolic.h"
#include "wubu_quaternion.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Flow Matching Configuration
 * =================================================================== */

typedef struct {
    int latent_dim;       /* dimension of each latent point (4 for quaternion) */
    int hidden_dim;       /* velocity network hidden dimension */
    int num_layers;       /* velocity network depth */
    int num_freqs;        /* positional encoding frequencies for time t */
    float sigma_min;      /* minimum noise for probability path */
    float sigma_max;      /* maximum noise for probability path */
    float learning_rate;  /* optimizer LR */
    int batch_size;       /* training batch size */
    int ode_steps;        /* number of ODE integration steps for inference */
} WubuFlowConfig;

/* ===================================================================
 * Velocity Network — MLP on Tangent Space
 * =================================================================== */

typedef struct {
    /* Weights */
    float* w1;            /* [hidden_dim, latent_dim + 2*num_freqs] — input: [PE(t), x_tangent] */
    float* b1;            /* [hidden_dim] */
    float* w2;            /* [hidden_dim, hidden_dim] */
    float* b2;            /* [hidden_dim] */
    float* w_out;         /* [latent_dim, hidden_dim] — output: tangent vector */
    float* b_out;         /* [latent_dim] */

    /* Config */
    int latent_dim;
    int hidden_dim;
    int num_freqs;
    int input_dim;        /* latent_dim + 2*num_freqs */
    bool initialized;
} WubuVelocityNet;

/* ===================================================================
 * Flow Matching Model
 * =================================================================== */

typedef struct {
    WubuVelocityNet velocity_net;
    WubuFlowConfig config;
    float c;              /* Poincaré ball curvature */
    int learn_curvature;  /* GAP-C013: FD-learn c during training */
    float log_c;          /* internal param, c = exp(log_c) */
    int step_count;
} WubuFlowMatching;

/* ===================================================================
 * Initialize / Free
 * =================================================================== */

int wubu_flow_init(WubuFlowMatching* model, const WubuFlowConfig* config, float c);
void wubu_flow_free(WubuFlowMatching* model);

/* ===================================================================
 * Probability Path — Geodesic Interpolation on Poincaré Ball
 * =================================================================== */

/* Compute geodesic interpolation: μ_t = exp_{x_0}(t · log_{x_0}(x_1)) */
void wubu_flow_geodesic_interpolate(float* mu_t, const float* x_0, const float* x_1,
                                     float t, int N, int D, float c);

/* GAP-C014: scale-aware variant — per-level scale `s` modulates the
 * exponential/log maps (WuBu Nesting's scale_aware maps), letting each
 * nesting level control how far tangent steps reach. */
void wubu_flow_geodesic_interpolate_scaled(float* mu_t, const float* x_0,
                                           const float* x_1, float t,
                                           int N, int D, float c, float scale);
/* Scale-aware target velocity companion. */
void wubu_flow_target_velocity_scaled(float* v_target, const float* x_0,
                                      const float* x_1, float t,
                                      int N, int D, float c, float scale);

/* Compute target velocity: d/dt μ_t at time t */
void wubu_flow_target_velocity(float* v_target, const float* x_0, const float* x_1,
                                float t, int N, int D, float c);

/* ===================================================================
 * Velocity Network Forward
 * =================================================================== */

/* Predict velocity at time t, position x (on tangent space at x) */
void wubu_flow_predict_velocity(WubuFlowMatching* model, float* v_pred,
                                 const float* x, float t, int N, int D);

/* ===================================================================
 * Training
 * =================================================================== */

/* Single training step: sample pairs (x_0, x_1) from key frame latents,
 * compute flow matching loss, update velocity network via SGD */
float wubu_flow_train_step(WubuFlowMatching* model,
                            const float* key_latents, int num_keyframes,
                            int points_per_frame);

/* ===================================================================
 * Inference — Generate Intermediate Frames
 * =================================================================== */

/* Integrator selection for the probability-flow ODE (GAP-C002) */
typedef enum {
    WUBU_ODE_EULER = 0,   /* x' = exp_x(h*v) via Mobius add          */
    WUBU_ODE_HEUN   = 1   /* predictor-corrector, 2nd order on ball  */
} WubuOdeSolver;

/* Generate intermediate frame between two key frames using ODE solve */
/* x_0: source latent [N, D], x_1: target latent [N, D] */
/* num_intermediate: number of frames to generate between key frames */
/* Output: [num_intermediate, N, D] generated latents */
float* wubu_flow_generate_intermediate(WubuFlowMatching* model,
                                        const float* x_0, const float* x_1,
                                        int N, int D,
                                        int num_intermediate);

/* GAP-C002: solver-selecting variant. solver = EULER | HEUN.
 * HEUN halves integration error at 2x velocity-net evaluations. */
float* wubu_flow_generate_intermediate_ex(WubuFlowMatching* model,
                                          const float* x_0, const float* x_1,
                                          int N, int D,
                                          int num_intermediate,
                                          WubuOdeSolver solver);

/* GAP-C008: P-frame residual coding. After flow prediction, the residual
 * r = x1 - pred is uniformly quantized to `levels` bins over [-mx,mx].
 * Returns malloc'd quantized residual [N*D] and writes the bits needed
 * (N*D*log2(levels)) to *out_bits. Decoder reconstructs pred+dequant(r). */
float* wubu_pframe_residual_encode(const float* pred, const float* x1,
                                    int N, int D, int levels,
                                    float* out_bits);

/* decode side: recon = pred + dequantized residual */
void wubu_pframe_residual_decode(float* recon, const float* pred,
                                  const float* qres, int N, int D,
                                  int levels);

/* GAP-C013: curvature learning toggle. */
void wubu_flow_set_learn_curvature(WubuFlowMatching* model, int enable);

/* GAP-C006: GRW-style tangent noise (Geodesic Random Walk recipe, node 2.4).
 * Samples gaussian noise in the tangent space at each point and maps it to
 * the manifold via exp map — the numerically-stable way to jitter latents
 * without leaving the ball. Returns count of points adjusted. */
void wubu_flow_tangent_noise(WubuFlowMatching* model, float* latents,
                              int N, int D, float sigma);

/* GAP-C005: projection-back gate. Any point with ||x||>=1 is rescaled to
 * r_max (0.999) radially; on-ball points untouched. Returns 1 if any
 * projection occurred (drift detected), 0 if all clean. */
int wubu_flow_project_back(float* latents, int N, int D, float r_max);

/* GAP-C009: multi-frame temporal rollout — walk keyframes k0..k_{M-1},
 * generating num_between intermediates on each leg (geodesic-chained).
 * Output: [(M-1)*num_between, N, D] latents. */
float* wubu_flow_rollout(WubuFlowMatching* model,
                         const float* key_latents, int M, int N, int D,
                         int num_between, WubuOdeSolver solver);

/* ===================================================================
 * Loss Computation
 * =================================================================== */

/* Flow matching loss: E[||v_θ(t, μ_t) - d/dt μ_t||²] */
float wubu_flow_compute_loss(WubuFlowMatching* model,
                              const float* x_0, const float* x_1,
                              int N, int D, float t);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FLOW_MATCHING_H */
