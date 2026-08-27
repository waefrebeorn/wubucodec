/*
 * wubu_hyperbolic.c -- Poincare Ball hyperbolic geometry
 *
 * Slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py (bytropix/draftPY/)
 * Faithful C11 implementation of HyperbolicUtils, Manifold, PoincareBall
 */

#include "wubu_hyperbolic.h"
#include <math.h>
#include <string.h>

/* ===================================================================
 * Internal helpers
 * =================================================================== */

static float vec_norm_sq(const float* x, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

static float vec_norm(const float* x, int N) {
    return sqrtf(vec_norm_sq(x, N));
}

static float vec_dot(const float* a, const float* b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* ===================================================================
 * Poincare Clip
 * =================================================================== */

void wubu_poincare_clip(float* out, const float* x, int N, float c) {
    if (c <= 0.0f) {
        /* Euclidean: just copy */
        memcpy(out, x, N * sizeof(float));
        return;
    }

    float sqrt_c = sqrtf(c + WUBU_EPS);
    float max_norm = 1.0f / (sqrt_c + WUBU_EPS);
    /* Slightly reduce radius to stay strictly inside */
    max_norm *= (1.0f - WUBU_EPS * 10.0f);

    float norm_sq = vec_norm_sq(x, N);
    float norm = sqrtf(norm_sq + WUBU_EPS);

    if (norm > max_norm) {
        float scale = max_norm / (norm + WUBU_EPS);
        for (int i = 0; i < N; i++) {
            out[i] = x[i] * scale;
        }
    } else {
        memcpy(out, x, N * sizeof(float));
    }
}

/* ===================================================================
 * Exponential Map: tangent space -> manifold
 * =================================================================== */

void wubu_expmap(float* out, const float* v, int N, float c) {
    if (c <= 0.0f) {
        /* Euclidean: identity */
        memcpy(out, v, N * sizeof(float));
        return;
    }

    float sqrt_c = sqrtf(c + WUBU_EPS);
    float v_norm = vec_norm(v, N);

    if (v_norm < WUBU_EPS) {
        /* Near origin: expmap_0(v) ≈ v */
        memcpy(out, v, N * sizeof(float));
        return;
    }

    float arg = sqrt_c * v_norm;
    /* Clamp tanh argument for stability */
    if (arg > 30.0f) arg = 30.0f;
    float tanh_val = tanhf(arg);
    float scale = tanh_val / (sqrt_c * v_norm + WUBU_EPS);

    for (int i = 0; i < N; i++) {
        out[i] = scale * v[i];
    }

    /* Final safety: clip to manifold */
    wubu_poincare_clip(out, out, N, c);
}

/* ===================================================================
 * Logarithmic Map: manifold -> tangent space
 * =================================================================== */

void wubu_logmap(float* out, const float* y, int N, float c) {
    if (c <= 0.0f) {
        /* Euclidean: identity */
        memcpy(out, y, N * sizeof(float));
        return;
    }

    /* Clip y to be strictly inside manifold first */
    float y_clipped[512]; /* max dim stack alloc */
    if (N > 512) {
        /* For large N, caller must provide buffer -- use heap */
        float* buf = (float*)malloc(N * sizeof(float));
        wubu_poincare_clip(buf, y, N, c);
        wubu_logmap(out, buf, N, c);
        free(buf);
        return;
    }
    wubu_poincare_clip(y_clipped, y, N, c);

    float sqrt_c = sqrtf(c + WUBU_EPS);
    float y_norm = vec_norm(y_clipped, N);

    if (y_norm < WUBU_EPS) {
        /* Near origin: logmap_0(y) ≈ y */
        memcpy(out, y_clipped, N * sizeof(float));
        return;
    }

    float atanh_arg = sqrt_c * y_norm;
    /* Clamp to be strictly within (-1, 1) for atanh */
    if (atanh_arg > 1.0f - WUBU_EPS * 4.0f) {
        atanh_arg = 1.0f - WUBU_EPS * 4.0f;
    }
    if (atanh_arg < -1.0f + WUBU_EPS * 4.0f) {
        atanh_arg = -1.0f + WUBU_EPS * 4.0f;
    }

    float atanh_val = atanhf(atanh_arg);
    float scale = atanh_val / (sqrt_c * y_norm + WUBU_EPS);

    for (int i = 0; i < N; i++) {
        out[i] = scale * y_clipped[i];
    }

    /* Clamp tangent vector magnitude for stability */
    float t_norm = vec_norm(out, N);
    if (t_norm > WUBU_TAN_VEC_CLAMP) {
        float clamp_scale = WUBU_TAN_VEC_CLAMP / t_norm;
        for (int i = 0; i < N; i++) {
            out[i] *= clamp_scale;
        }
    }
}

/* ===================================================================
 * Scale-Aware Exponential Map
 * =================================================================== */

void wubu_expmap_scaled(float* out, const float* v, int N, float c, float scale) {
    if (c <= 0.0f) {
        /* Euclidean: just scale */
        for (int i = 0; i < N; i++) {
            out[i] = scale * v[i];
        }
        return;
    }

    float sqrt_c = sqrtf(c + WUBU_EPS);
    float v_norm = vec_norm(v, N);

    if (v_norm < WUBU_EPS) {
        for (int i = 0; i < N; i++) {
            out[i] = scale * v[i];
        }
        return;
    }

    float arg = scale * sqrt_c * v_norm;
    if (arg > 30.0f) arg = 30.0f;
    float tanh_val = tanhf(arg);
    float s = tanh_val / (sqrt_c * v_norm + WUBU_EPS);

    for (int i = 0; i < N; i++) {
        out[i] = s * v[i];
    }

    wubu_poincare_clip(out, out, N, c);
}

/* ===================================================================
 * Scale-Aware Logarithmic Map
 * =================================================================== */

void wubu_logmap_scaled(float* out, const float* y, int N, float c, float scale) {
    if (c <= 0.0f) {
        for (int i = 0; i < N; i++) {
            out[i] = y[i] / fmaxf(scale, WUBU_EPS);
        }
        return;
    }

    float y_clipped[512];
    if (N > 512) {
        float* buf = (float*)malloc(N * sizeof(float));
        wubu_poincare_clip(buf, y, N, c);
        wubu_logmap_scaled(out, buf, N, c, scale);
        free(buf);
        return;
    }
    wubu_poincare_clip(y_clipped, y, N, c);

    float sqrt_c = sqrtf(c + WUBU_EPS);
    float y_norm = vec_norm(y_clipped, N);

    if (y_norm < WUBU_EPS) {
        for (int i = 0; i < N; i++) {
            out[i] = y_clipped[i] / fmaxf(scale, WUBU_EPS);
        }
        return;
    }

    float atanh_arg = scale * sqrt_c * y_norm;
    if (atanh_arg > 1.0f - WUBU_EPS * 4.0f) {
        atanh_arg = 1.0f - WUBU_EPS * 4.0f;
    }
    float atanh_val = atanhf(atanh_arg);
    float s = atanh_val / (scale * sqrt_c * y_norm + WUBU_EPS);

    for (int i = 0; i < N; i++) {
        out[i] = s * y_clipped[i];
    }

    float t_norm = vec_norm(out, N);
    if (t_norm > WUBU_TAN_VEC_CLAMP) {
        float clamp_scale = WUBU_TAN_VEC_CLAMP / t_norm;
        for (int i = 0; i < N; i++) {
            out[i] *= clamp_scale;
        }
    }
}

/* ===================================================================
 * Mobius Addition
 * =================================================================== */

void wubu_mobius_add(float* out, const float* x, const float* y, int N, float c) {
    if (c <= 0.0f) {
        /* Euclidean addition */
        for (int i = 0; i < N; i++) {
            out[i] = x[i] + y[i];
        }
        return;
    }

    float x_norm_sq = vec_norm_sq(x, N);
    float y_norm_sq = vec_norm_sq(y, N);
    float xy_dot = vec_dot(x, y, N);

    /* Clamp norms to stay within radius */
    float radius_sq = 1.0f / (c + WUBU_EPS);
    if (x_norm_sq > radius_sq * (1.0f - WUBU_EPS) * (1.0f - WUBU_EPS)) {
        x_norm_sq = radius_sq * (1.0f - WUBU_EPS) * (1.0f - WUBU_EPS);
    }
    if (y_norm_sq > radius_sq * (1.0f - WUBU_EPS) * (1.0f - WUBU_EPS)) {
        y_norm_sq = radius_sq * (1.0f - WUBU_EPS) * (1.0f - WUBU_EPS);
    }

    float denom = 1.0f + 2.0f * c * xy_dot + c * c * x_norm_sq * y_norm_sq;
    if (denom < WUBU_EPS) denom = WUBU_EPS;

    float num_x = 1.0f + 2.0f * c * xy_dot + c * y_norm_sq;
    float num_y = 1.0f - c * x_norm_sq;

    for (int i = 0; i < N; i++) {
        out[i] = (num_x * x[i] + num_y * y[i]) / denom;
    }

    /* Project result back onto manifold */
    wubu_poincare_clip(out, out, N, c);
}

/* ===================================================================
 * Euclidean to Riemannian Gradient
 * =================================================================== */

void wubu_egrad2rgrad(float* out, const float* p, const float* egrad, int N, float c) {
    if (c <= 0.0f) {
        memcpy(out, egrad, N * sizeof(float));
        return;
    }

    float p_norm_sq = vec_norm_sq(p, N);
    float max_norm_sq = 1.0f / (c + WUBU_EPS);
    /* Clamp to stay within radius */
    if (p_norm_sq > max_norm_sq * (1.0f - WUBU_EPS * 10.0f) * (1.0f - WUBU_EPS * 10.0f)) {
        p_norm_sq = max_norm_sq * (1.0f - WUBU_EPS * 10.0f) * (1.0f - WUBU_EPS * 10.0f);
    }

    float factor = (1.0f - c * p_norm_sq) / 2.0f;
    factor = factor * factor;
    /* Clamp minimum for stability */
    if (factor < WUBU_EPS * WUBU_EPS) factor = WUBU_EPS * WUBU_EPS;

    for (int i = 0; i < N; i++) {
        out[i] = factor * egrad[i];
    }
}

/* ===================================================================
 * Hyperbolic Distance
 * =================================================================== */

float wubu_hyperbolic_distance(const float* x, const float* y, int N, float c) {
    if (c <= 0.0f) {
        /* Euclidean distance */
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float d = x[i] - y[i];
            sum += d * d;
        }
        return sqrtf(sum);
    }

    /* Compute (-x) (+) y first */
    float* neg_x = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];

    float* mobius_result = (float*)malloc((size_t)N * sizeof(float));
    wubu_mobius_add(mobius_result, neg_x, y, N, c);

    float m_norm = vec_norm(mobius_result, N);
    float sqrt_c = sqrtf(c + WUBU_EPS);

    float arg = sqrt_c * m_norm;
    if (arg > 1.0f - WUBU_EPS * 4.0f) {
        arg = 1.0f - WUBU_EPS * 4.0f;
    }

    free(neg_x);
    free(mobius_result);
    return (2.0f / sqrt_c) * atanhf(arg);
}
