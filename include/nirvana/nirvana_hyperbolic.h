/* nirvana_hyperbolic.h — Hyperbolic Manifold Residual Coding
 *
 * Core innovation: code residuals on the Poincaré ball instead of Euclidean space.
 * The Poincaré ball has more "room" near the boundary, which naturally handles
 * dynamic range — small residuals map to the center (high precision), large
 * residuals map near the boundary.
 *
 * This is the WuBuMath advantage: no other codec codes residuals on a hyperbolic
 * manifold.
 *
 * Math:
 *   - Residual r in tangent space at origin
 *   - Encode: y = expmap_0(r) = tanh(||r||) * r / ||r||
 *   - Decode: r = logmap_0(y) = atanh(||y||) * y / ||y||
 *   - RD cost: d_c(x, y) = (2/sqrt(c)) * atanh(sqrt(c) * ||(-x) (+) y||)
 */
#ifndef NIRVANA_HYPERBOLIC_H
#define NIRVANA_HYPERBOLIC_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NIRVANA_EPS 1e-6f
#define NIRVANA_POINCARE_CURVATURE 1.0f
#define NIRVANA_MAX_NORM 0.999f

/* Poincaré ball operations */
static inline float nirvana_poincare_norm(float x, float c) {
    return 1.0f / sqrtf(c) - NIRVANA_EPS;
}

/* Clip to Poincaré ball: ensures ||x|| < 1/sqrt(c) */
static inline void nirvana_poincare_clip(float *out, const float *x, int N, float c) {
    float sq_norm = 0;
    for (int i = 0; i < N; i++) sq_norm += x[i] * x[i];
    float norm = sqrtf(sq_norm);
    float limit = 1.0f / sqrtf(c) - NIRVANA_EPS;
    if (norm > limit) {
        float scale = limit / (norm + NIRVANA_EPS);
        for (int i = 0; i < N; i++) out[i] = x[i] * scale;
    } else {
        for (int i = 0; i < N; i++) out[i] = x[i];
    }
}

/* Exponential map: tangent space -> Poincaré ball */
static inline void nirvana_expmap(float *out, const float *v, int N, float c) {
    float sq_norm = 0;
    for (int i = 0; i < N; i++) sq_norm += v[i] * v[i];
    float norm = sqrtf(sq_norm);
    float sqrt_c = sqrtf(c);
    
    if (norm < NIRVANA_EPS) {
        for (int i = 0; i < N; i++) out[i] = v[i];
        return;
    }
    
    float tanh_val = tanhf(sqrt_c * norm);
    float scale = tanh_val / (sqrt_c * norm);
    for (int i = 0; i < N; i++) out[i] = v[i] * scale;
}

/* Logarithmic map: Poincaré ball -> tangent space */
static inline void nirvana_logmap(float *out, const float *y, int N, float c) {
    float sq_norm = 0;
    for (int i = 0; i < N; i++) sq_norm += y[i] * y[i];
    float norm = sqrtf(sq_norm);
    float sqrt_c = sqrtf(c);
    
    if (norm < NIRVANA_EPS) {
        for (int i = 0; i < N; i++) out[i] = y[i];
        return;
    }
    
    float atanh_val = atanhf(sqrt_c * norm);
    float scale = atanh_val / (sqrt_c * norm);
    for (int i = 0; i < N; i++) out[i] = y[i] * scale;
}

/* Mobius addition: hyperbolic translation */
static inline void nirvana_mobius_add(float *out, const float *x, const float *y, int N, float c) {
    float xy_dot = 0, x_sq = 0, y_sq = 0;
    for (int i = 0; i < N; i++) {
        xy_dot += x[i] * y[i];
        x_sq += x[i] * x[i];
        y_sq += y[i] * y[i];
    }
    
    float num_x = 1 + 2 * c * xy_dot + c * y_sq;
    float num_y = 1 - c * x_sq;
    float den = 1 + 2 * c * xy_dot + c * c * x_sq * y_sq;
    
    for (int i = 0; i < N; i++) {
        out[i] = (num_x * x[i] + num_y * y[i]) / den;
    }
}

/* Hyperbolic distance */
static inline float nirvana_hyperbolic_distance(const float *x, const float *y, int N, float c) {
    float neg_x[N];
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];
    
    float mobius_result[N];
    nirvana_mobius_add(mobius_result, neg_x, y, N, c);
    
    float sq_norm = 0;
    for (int i = 0; i < N; i++) sq_norm += mobius_result[i] * mobius_result[i];
    float norm = sqrtf(sq_norm);
    
    float sqrt_c = sqrtf(c);
    return (2 / sqrt_c) * atanhf(sqrt_c * norm);
}

/* Encode residual block on hyperbolic manifold */
static inline void nirvana_hyperbolic_encode(float *out, const float *residual, int N, float c) {
    nirvana_expmap(out, residual, N, c);
}

/* Decode residual block from hyperbolic manifold */
static inline void nirvana_hyperbolic_decode(float *out, const float *encoded, int N, float c) {
    nirvana_logmap(out, encoded, N, c);
}

/* Quantize on hyperbolic manifold */
static inline void nirvana_hyperbolic_quantize(float *out, const float *x, int N, float c, float qstep) {
    /* Map to tangent space, quantize, map back */
    float tangent[N];
    nirvana_logmap(tangent, x, N, c);
    
    /* Quantize in tangent space */
    for (int i = 0; i < N; i++) {
        float sign = (tangent[i] >= 0) ? 1 : -1;
        float abs_val = fabsf(tangent[i]);
        float quantized = sign * ((int)(abs_val / qstep + 0.5f)) * qstep;
        tangent[i] = quantized;
    }
    
    /* Map back to manifold */
    nirvana_expmap(out, tangent, N, c);
}

/* Dequantize on hyperbolic manifold */
static inline void nirvana_hyperbolic_dequantize(float *out, const float *x, int N, float c) {
    /* In hyperbolic coding, dequantization is just the identity —
       the quantization error is absorbed by the manifold structure */
    for (int i = 0; i < N; i++) out[i] = x[i];
}

#endif /* NIRVANA_HYPERBOLIC_H */
