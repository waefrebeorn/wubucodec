/*
 * wubu_tangent_flow.c -- Tangent flow transformations
 *
 * Slermed from HyperbolicWuBuNestingLevel (WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * Implements learned residual transformations in tangent space.
 */

#include "wubu_tangent_flow.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ===================================================================
 * Initialize
 * =================================================================== */

int wubu_tangent_flow_init(WubuTangentFlow* flow, const WubuTangentFlowConfig* config) {
    flow->dim = config->input_dim;
    flow->hidden_dim = config->hidden_dim;
    flow->flow_type = config->flow_type;
    flow->dropout = config->dropout;
    flow->initialized = true;

    /* Initialize scale: softplus(initial_scale) */
    float val = config->initial_scale;
    if (val < 1e-6f) val = 1e-6f;
    /* Inverse softplus: log(exp(x) - 1) */
    flow->log_scale_raw = logf(expf(val) - 1.0f);
    flow->current_scale = val;

    int d = flow->dim;
    int h = flow->hidden_dim;

    if (flow->flow_type == 0) {
        /* MLP flow: w_gate [h, d], w1 [h, d], w2 [d, h], bias1 [h], bias2 [d] */
        flow->w_gate = (float*)calloc(h * d, sizeof(float));
        flow->w1 = (float*)calloc(h * d, sizeof(float));
        flow->w2 = (float*)calloc(d * h, sizeof(float));
        flow->bias1 = (float*)calloc(h, sizeof(float));
        flow->bias2 = (float*)calloc(d, sizeof(float));

        /* Xavier init */
        float limit_gate = sqrtf(6.0f / (float)(d + h));
        float limit_1 = sqrtf(6.0f / (float)(d + h));
        float limit_2 = sqrtf(6.0f / (float)(h + d));
        for (int i = 0; i < h * d; i++) {
            flow->w_gate[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit_gate;
            flow->w1[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit_1;
        }
        for (int i = 0; i < d * h; i++) {
            flow->w2[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit_2;
        }
    } else {
        /* Linear flow: w1 [d, d], bias1 [d] */
        flow->w_gate = NULL;
        flow->w1 = (float*)calloc(d * d, sizeof(float));
        flow->w2 = NULL;
        flow->bias1 = (float*)calloc(d, sizeof(float));
        flow->bias2 = NULL;

        /* Initialize as identity */
        for (int i = 0; i < d; i++) {
            flow->w1[i * d + i] = 1.0f;
        }
    }

    return 0;
}

/* ===================================================================
 * Free
 * =================================================================== */

void wubu_tangent_flow_free(WubuTangentFlow* flow) {
    free(flow->w1); flow->w1 = NULL;
    free(flow->w2); flow->w2 = NULL;
    free(flow->w_gate); flow->w_gate = NULL;
    free(flow->bias1); flow->bias1 = NULL;
    free(flow->bias2); flow->bias2 = NULL;
    flow->initialized = false;
}

/* ===================================================================
 * SiLU activation (SwiGLU component)
 * =================================================================== */

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ===================================================================
 * Forward pass
 * =================================================================== */

void wubu_tangent_flow_forward(WubuTangentFlow* flow, const float* x, float* out, int N) {
    if (!flow->initialized || N != flow->dim) return;

    int d = flow->dim;
    int h = flow->hidden_dim;
    float scale = flow->current_scale;

    if (flow->flow_type == 0) {
        /* MLP with SwiGLU: out = w2 @ (silu(w_gate @ x + bias1) * (w1 @ x)) + bias2 */
        float hidden[256]; /* max hidden dim */
        float gate[256];

        for (int j = 0; j < h; j++) {
            float g = flow->bias1[j];
            float a = 0.0f;
            for (int i = 0; i < d; i++) {
                g += flow->w_gate[j * d + i] * x[i];
                a += flow->w1[j * d + i] * x[i];
            }
            gate[j] = silu(g);
            hidden[j] = gate[j] * a;
        }

        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < h; j++) {
                val += flow->w2[i * h + j] * hidden[j];
            }
            out[i] = scale * val;
        }
    } else {
        /* Linear: out = scale * (w1 @ x + bias1) */
        for (int i = 0; i < d; i++) {
            float val = flow->bias1[i];
            for (int j = 0; j < d; j++) {
                val += flow->w1[i * d + j] * x[j];
            }
            out[i] = scale * val;
        }
    }
}

/* ===================================================================
 * Get scale
 * =================================================================== */

float wubu_tangent_flow_get_scale(const WubuTangentFlow* flow) {
    return flow->current_scale;
}
