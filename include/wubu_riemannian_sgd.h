/*
 * wubu_riemannian_sgd.h -- Riemannian Enhanced SGD Optimizer
 *
 * Slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py (bytropix/draftPY/)
 * Faithful C11 translation of RiemannianEnhancedSGD
 *
 * Features:
 *   - Momentum buffer per parameter
 *   - Weight decay (L2 regularization)
 *   - Gradient clipping (max norm)
 *   - Hyperbolic parameter updates via exponential map
 *   - Euclidean parameter updates with standard SGD
 *   - Automatic fallback on numerical instability
 */

#ifndef WUBU_RIEMANNIAN_SGD_H
#define WUBU_RIEMANNIAN_SGD_H

#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Constants
 * =================================================================== */

#define WUBU_SGD_EPS 1e-6f
#define WUBU_SGD_MAX_GRAD_NORM 1.0f
#define WUBU_SGD_EUCLIDEAN_CLAMP 1e6f

/* ===================================================================
 * Parameter manifold binding
 * =================================================================== */

typedef struct {
    float c;          /* curvature: >0 = hyperbolic, <=0 = Euclidean */
    int manifold_enabled; /* 1 = use manifold operations, 0 = Euclidean */
} WubuManifoldBinding;

/* ===================================================================
 * Per-parameter state (momentum buffer)
 * =================================================================== */

typedef struct {
    float* momentum;   /* [N] momentum buffer */
    int N;             /* parameter count */
    bool initialized;
} WubuParamState;

/* ===================================================================
 * Optimizer config
 * =================================================================== */

typedef struct {
    float learning_rate;
    float initial_lr;
    float momentum_factor;
    float initial_momentum;
    float weight_decay;
    float max_grad_norm;
    bool q_controller_enabled;
} WubuSGDConfig;

/* ===================================================================
 * Optimizer state (one per parameter group)
 * =================================================================== */

typedef struct {
    WubuSGDConfig config;
    WubuManifoldBinding manifold;
    WubuParamState param_state;
    float current_lr;
    float current_momentum;
    int step_count;
} WubuSGD;

/* ===================================================================
 * Initialize optimizer
 * =================================================================== */

void wubu_sgd_init(WubuSGD* opt, const WubuSGDConfig* config, WubuManifoldBinding manifold, int param_count);

/* ===================================================================
 * Free optimizer resources
 * =================================================================== */

void wubu_sgd_free(WubuSGD* opt);

/* ===================================================================
 * SGD step (Euclidean parameter)
 * =================================================================== */
void wubu_sgd_step_euclidean(WubuSGD* opt, float* param, const float* grad, int N);

/* ===================================================================
 * SGD step (Hyperbolic parameter via Poincare ball)
 * =================================================================== */

void wubu_sgd_step_hyperbolic(WubuSGD* opt, float* param, const float* grad, int N);

/* ===================================================================
 * Zero momentum buffer
 * =================================================================== */

void wubu_sgd_zero_momentum(WubuSGD* opt);

/* ===================================================================
 * Update learning rate (called by Q-controller)
 * =================================================================== */

void wubu_sgd_set_lr(WubuSGD* opt, float lr);
void wubu_sgd_set_momentum(WubuSGD* opt, float momentum);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RIEMANNIAN_SGD_H */
