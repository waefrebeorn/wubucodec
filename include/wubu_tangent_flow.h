/*
 * wubu_tangent_flow.h -- Tangent flow transformations
 *
 * Slermed from HyperbolicWuBuNestingLevel (WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * Tangent flow applies a learned residual transformation in tangent space
 * before exponential mapping to the next manifold level.
 *
 * Features:
 *   - MLP-based tangent flow (SwiGLU activation)
 *   - Linear tangent flow (simple projection)
 *   - Learnable scale factor (softplus-constrained)
 */

#ifndef WUBU_TANGENT_FLOW_H
#define WUBU_TANGENT_FLOW_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Tangent Flow Configuration
 * =================================================================== */

typedef struct {
    int input_dim;          /* input tangent dimension */
    int hidden_dim;         /* hidden layer dimension */
    float dropout;          /* dropout probability */
    int flow_type;          /* 0 = MLP (SwiGLU), 1 = Linear */
    float initial_scale;    /* initial learnable scale */
} WubuTangentFlowConfig;

/* ===================================================================
 * Tangent Flow State (weights + buffers)
 * =================================================================== */

typedef struct {
    /* Weights */
    float* w1;              /* [hidden_dim, input_dim] */
    float* w2;              /* [input_dim, hidden_dim] */
    float* w_gate;          /* [hidden_dim, input_dim] for SwiGLU */
    float* bias1;           /* [hidden_dim] */
    float* bias2;           /* [input_dim] */

    /* Learnable scale (softplus-constrained) */
    float log_scale_raw;    /* raw unconstrained scale parameter */
    float current_scale;    /* computed scale value */

    /* Dimensions */
    int dim;
    int hidden_dim;
    int flow_type;
    float dropout;
    bool initialized;
} WubuTangentFlow;

/* ===================================================================
 * Initialize tangent flow
 * =================================================================== */

int wubu_tangent_flow_init(WubuTangentFlow* flow, const WubuTangentFlowConfig* config);

/* ===================================================================
 * Free tangent flow resources
 * =================================================================== */

void wubu_tangent_flow_free(WubuTangentFlow* flow);

/* ===================================================================
 * Forward pass through tangent flow
 * ===================================================================
 * Input:  x [dim] -- input tangent vector
 * Output: out [dim] -- transformed tangent vector (scaled by learnable scale)
 */

void wubu_tangent_flow_forward(WubuTangentFlow* flow, const float* x, float* out, int N);

/* ===================================================================
 * Get current scale value
 * =================================================================== */

float wubu_tangent_flow_get_scale(const WubuTangentFlow* flow);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TANGENT_FLOW_H */
