/*
 * wubu_parallel_transport.c -- Parallel transport on Poincare ball
 *
 * Slermed from FullyHyperbolicWuBuNestingModel (WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * Implements parallel transport using the gyration-based formula.
 */

#include "wubu_parallel_transport.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ===================================================================
 * Lambda factor: lambda_x = 2 / (1 - c * ||x||^2)
 * =================================================================== */

float wubu_lambda_factor(const float* x, int N, float c) {
    if (c <= 0.0f) return 1.0f;

    float norm_sq = 0.0f;
    for (int i = 0; i < N; i++) norm_sq += x[i] * x[i];

    float denom = 1.0f - c * norm_sq;
    /* Clamp to avoid division by zero at boundary */
    if (denom < WUBU_EPS * 100.0f) denom = WUBU_EPS * 100.0f;

    return 2.0f / denom;
}

/* ===================================================================
 * Parallel transport: origin -> p
 * ===================================================================
 * P_0->p(v) = (lambda_0 / lambda_p) * v
 * lambda_0 = 2 / (1 - 0) = 2 (since ||0|| = 0)
 * So P_0->p(v) = (2 / lambda_p) * v = (1 - c*||p||^2) * v
 * =================================================================== */

void wubu_parallel_transport_to_p(float* out, const float* v, const float* p, int N, float c) {
    if (c <= 0.0f) {
        memcpy(out, v, N * sizeof(float));
        return;
    }

    float lp = wubu_lambda_factor(p, N, c);
    /* lambda_0 = 2, so scale = 2 / lp = 1 - c*||p||^2 */
    float scale = 2.0f / lp;

    for (int i = 0; i < N; i++) {
        out[i] = scale * v[i];
    }
}

/* ===================================================================
 * Parallel transport: p -> origin
 * ===================================================================
 * P_p->0(v) = (lambda_p / lambda_0) * v = (lambda_p / 2) * v
 * =================================================================== */

void wubu_parallel_transport_to_origin(float* out, const float* v, const float* p, int N, float c) {
    if (c <= 0.0f) {
        memcpy(out, v, N * sizeof(float));
        return;
    }

    float lp = wubu_lambda_factor(p, N, c);
    float scale = lp / 2.0f;

    for (int i = 0; i < N; i++) {
        out[i] = scale * v[i];
    }
}

/* ===================================================================
 * Parallel transport: p -> q (via origin)
 * =================================================================== */

void wubu_parallel_transport(float* out, const float* v, const float* p, const float* q, int N, float c) {
    if (c <= 0.0f) {
        memcpy(out, v, N * sizeof(float));
        return;
    }

    /* First transport p -> origin */
    float v_at_origin[512];
    bool heap = (N > 512);
    float* v_o;
    if (heap) v_o = (float*)malloc(N * sizeof(float));
    else v_o = v_at_origin;

    wubu_parallel_transport_to_origin(v_o, v, p, N, c);

    /* Then transport origin -> q */
    wubu_parallel_transport_to_p(out, v_o, q, N, c);

    if (heap) free(v_o);
}
