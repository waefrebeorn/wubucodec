/*
 * wubu_quaternion_ops.c -- Hamilton Quaternion Operations for WuBu
 *
 * Implements the proper WUBU math:
 *   - Hamilton product: p * q
 *   - Quaternion conjugate, normalize, inverse
 *   - Quaternion rotation: p * v * q
 *   - SLERP (spherical linear interpolation) for smooth rotations
 *   - Conversion between quaternion and rotation matrix
 */

#include "wubu_hyperbolic.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===================================================================
 * Basic Quaternion Operations
 * =================================================================== */

/* Hamilton product: p * q
 * q = (w, x, y, z)
 *
 * w = p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z
 * x = p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y
 * y = p.w*q.y - p.x*q.z + p.y*q.w + p.z*q.x
 * z = p.w*q.z + p.x*q.y - p.y*q.x + p.z*q.w
 */
void wubu_hamilton_product(float* out, const float* p, const float* q) {
    out[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3];  /* w */
    out[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2];  /* x */
    out[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1];  /* y */
    out[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0];  /* z */
}

/* Hamilton product for batched quaternions [N, 4] */
void wubu_hamilton_product_batch(float* out, const float* p, const float* q, int N) {
    for (int i = 0; i < N; i++) {
        wubu_hamilton_product(out + i * 4, p + i * 4, q + i * 4);
    }
}

/* Quaternion conjugate: q* = (w, -x, -y, -z) */
void wubu_quat_conjugate(float* out, const float* q) {
    out[0] = q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

/* Quaternion norm squared */
float wubu_quat_norm_sq(const float* q) {
    return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
}

/* Quaternion norm */
float wubu_quat_norm(const float* q) {
    return sqrtf(wubu_quat_norm_sq(q));
}

/* Normalize quaternion to unit length */
void wubu_quat_normalize(float* out, const float* q) {
    float norm = wubu_quat_norm(q);
    if (norm < 1e-8f) {
        out[0] = 1.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
        return;
    }
    float inv = 1.0f / norm;
    out[0] = q[0] * inv;
    out[1] = q[1] * inv;
    out[2] = q[2] * inv;
    out[3] = q[3] * inv;
}

/* Unit quaternion check (|q| ≈ 1) */
int wubu_quat_is_unit(const float* q) {
    float nsq = wubu_quat_norm_sq(q);
    return (nsq > 0.99f && nsq < 1.01f);
}

/* ===================================================================
 * Quaternion Rotation: R(v) = p * v * q
 *
 * This is the core WUBU operation. For a vector v encoded as a pure
 * quaternion (0, vx, vy, vz), rotation by unit quaternion p gives:
 *   R(v) = p * (0, v) * p*
 *
 * For the dual-quaternion form p * v * q (used in TangentSpaceRotation):
 *   R(v) = p * (0, v) * q
 * =================================================================== */

/* Rotate pure vector v (treated as quaternion (0, vx, vy, vz))
 * by unit quaternion p: R(v) = p * v * p* */
void wubu_quat_rotate_vector(float* out_vec, const float* p, const float* v) {
    /* v as pure quaternion: (0, vx, vy, vz) */
    float vq[4] = {0.0f, v[0], v[1], v[2]};

    /* p * v */
    float pv[4];
    wubu_hamilton_product(pv, p, vq);

    /* (p * v) * p* */
    float p_conj[4];
    wubu_quat_conjugate(p_conj, p);

    float result[4];
    wubu_hamilton_product(result, pv, p_conj);

    /* Extract vector part (result[0] should be ~0) */
    out_vec[0] = result[1];
    out_vec[1] = result[2];
    out_vec[2] = result[3];
}

/* Dual-quaternion rotation: R(v) = p * v * q
 * Used in TangentSpaceRotation for inter-level transitions.
 * p and q must be unit quaternions. */
void wubu_quat_rotate_dual(float* out_vec, const float* p, const float* q, const float* v) {
    /* v as pure quaternion */
    float vq[4] = {0.0f, v[0], v[1], v[2]};

    /* p * v */
    float pv[4];
    wubu_hamilton_product(pv, p, vq);

    /* (p * v) * q */
    float result[4];
    wubu_hamilton_product(result, pv, q);

    out_vec[0] = result[1];
    out_vec[1] = result[2];
    out_vec[2] = result[3];
}

/* Batched rotation: rotate N vectors by corresponding N quaternions */
void wubu_quat_rotate_batch(float* out, const float* p, const float* v, int N) {
    for (int i = 0; i < N; i++) {
        wubu_quat_rotate_vector(out + i * 3, p + i * 4, v + i * 3);
    }
}

/* ===================================================================
 * SLERP: Spherical Linear Interpolation
 *
 * Interpolates between two unit quaternions p0 and p1 by parameter t.
 * Used for smooth rotation between key frames in flow matching.
 *
 * slerp(p0, p1, t) = p0 * sin((1-t)*theta)/sin(theta) + p1 * sin(t*theta)/sin(theta)
 * where theta = acos(<p0, p1>)
 * =================================================================== */

void wubu_quat_slerp(float* out, const float* p0, const float* p1, float t) {
    /* Compute dot product */
    float dot = p0[0]*p1[0] + p0[1]*p1[1] + p0[2]*p1[2] + p0[3]*p1[3];

    /* If dot < 0, negate one quaternion to take shorter arc */
    float p1_adj[4];
    if (dot < 0.0f) {
        p1_adj[0] = -p1[0]; p1_adj[1] = -p1[1];
        p1_adj[2] = -p1[2]; p1_adj[3] = -p1[3];
        dot = -dot;
    } else {
        p1_adj[0] = p1[0]; p1_adj[1] = p1[1];
        p1_adj[2] = p1[2]; p1_adj[3] = p1[3];
    }

    float dot_clamped = fminf(1.0f, fmaxf(-1.0f, dot));

    if (dot_clamped > 0.9995f) {
        /* Very close quaternions: use linear interpolation */
        out[0] = p0[0] + t * (p1_adj[0] - p0[0]);
        out[1] = p0[1] + t * (p1_adj[1] - p0[1]);
        out[2] = p0[2] + t * (p1_adj[2] - p0[2]);
        out[3] = p0[3] + t * (p1_adj[3] - p0[3]);
    } else {
        float theta = acosf(dot_clamped);
        float sin_theta = sinf(theta);
        float a = sinf((1.0f - t) * theta) / sin_theta;
        float b = sinf(t * theta) / sin_theta;

        out[0] = a * p0[0] + b * p1_adj[0];
        out[1] = a * p0[1] + b * p1_adj[1];
        out[2] = a * p0[2] + b * p1_adj[2];
        out[3] = a * p0[3] + b * p1_adj[3];
    }

    /* Normalize result */
    wubu_quat_normalize(out, out);
}

/* ===================================================================
 * Quaternion Exponential / Logarithmic Map
 *
 * exp: tangent vector → unit quaternion (for optimization)
 * log: unit quaternion → tangent vector
 * =================================================================== */

/* Quaternion exponential: exp(v) for pure quaternion v = (0, vx, vy, vz)
 * exp(v) = (cos(||v||), sin(||v||) * v/||v||)
 * GAP-H008 FIX: v is now read as a pure QUATERNION (w ignored, vector in
 * v[1..3]) matching wubu_quat_log's output convention. Previously exp read
 * the vector from v[0..2], making log(exp(v)) shift components — caught by
 * this property gate. */
void wubu_quat_exp(float* out, const float* v) {
    float vnorm = sqrtf(v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
    if (vnorm < 1e-8f) {
        out[0] = 1.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
        return;
    }
    float s = sinf(vnorm) / vnorm;
    out[0] = cosf(vnorm);
    out[1] = v[1] * s;
    out[2] = v[2] * s;
    out[3] = v[3] * s;
}

/* Quaternion logarithm: log(q) for unit quaternion q
 * log(q) = (0, theta/sin(theta) * (x, y, z)) where theta = acos(w)
 */
void wubu_quat_log(float* out, const float* q) {
    float w_clamped = fminf(1.0f, fmaxf(-1.0f, q[0]));
    float theta = acosf(w_clamped);
    float sin_theta = sinf(theta);

    if (sin_theta < 1e-8f) {
        out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 0.0f;
        return;
    }
    float scale = theta / sin_theta;
    out[0] = 0.0f;
    out[1] = q[1] * scale;
    out[2] = q[2] * scale;
    out[3] = q[3] * scale;
}

/* ===================================================================
 * Poincaré Ball Operations (Hyperbolic Geometry)
 *
 * These implement the hyperbolic geometry that makes WuBu special.
 * The Poincaré ball model maps hyperbolic space to the unit ball.
 * =================================================================== */

/* Poincaré exponential map at origin:
 * exp_0^c(v) = tanh(√c·||v||) · v / (√c·||v||)
 */
void wubu_poincare_exp_map(float* out, const float* v, int D, float c) {
    if (c <= 0.0f) { memcpy(out, v, (size_t)D * sizeof(float)); return; }
    float sqrt_c = sqrtf(c);
    float vnorm = 0.0f;
    for (int d = 0; d < D; d++) vnorm += v[d] * v[d];
    vnorm = sqrtf(vnorm);
    if (vnorm < 1e-8f) { memcpy(out, v, (size_t)D * sizeof(float)); return; }
    float arg = fminf(sqrt_c * vnorm, 30.0f);
    float scale = tanhf(arg) / (sqrt_c * vnorm);
    for (int d = 0; d < D; d++) out[d] = scale * v[d];
}

/* Poincaré logarithmic map at origin:
 * log_0^c(y) = atanh(√c·||y||) · y / (√c·||y||)
 */
void wubu_poincare_log_map(float* out, const float* y, int D, float c) {
    if (c <= 0.0f) { memcpy(out, y, (size_t)D * sizeof(float)); return; }
    float ynorm = 0.0f;
    for (int d = 0; d < D; d++) ynorm += y[d] * y[d];
    ynorm = sqrtf(ynorm);
    float sqrt_c = sqrtf(c);
    float arg = fminf(sqrt_c * ynorm, 1.0f - 1e-6f);
    float scale = atanhf(arg) / (sqrt_c * ynorm + 1e-8f);
    for (int d = 0; d < D; d++) out[d] = y[d] * scale;
}

/* Möbius addition: x ⊕_c y
 * The hyperbolic analog of vector addition.
 * NOTE: This is defined in wubu_hyperbolic.c to avoid duplication.
 * The implementation there uses the standard gyrovector formula.
 */

/* ===================================================================
 * WuBu Nesting: Inter-Level Tangent Space Rotation
 *
 * The core operation that makes WuBu special:
 *   1. Log map at current level: project to tangent space
 *   2. Hamilton rotation: apply learned SO(4) rotation
 *   3. Exp map at next level: project to next hyperbolic space
 * =================================================================== */

/* Inter-level transition: rotate in tangent space, then exp map to next level */
void wubu_nesting_transition(float* out, const float* pos, int D,
                              const float* rot_p, const float* rot_q,
                              float c_current, float c_next) {
    /* 1. Log map: current hyperbolic → tangent space */
    float tangent[64]; /* max dim */
    wubu_poincare_log_map(tangent, pos, D, c_current);

    /* 2. Rotate in tangent space using Hamilton product */
    float rotated[64];
    wubu_quat_rotate_dual(rotated, rot_p, rot_q, tangent);

    /* 3. Exp map: tangent space → next hyperbolic level */
    wubu_poincare_exp_map(out, rotated, D, c_next);
}

/* ===================================================================
 * Compression-Optimized Encode/Decode
 *
 * Uses Hamilton rotation to encode RGB into a rotation that can be
 * perfectly decoded (within quantization precision).
 *
 * Encode: RGB → rotation quaternion q such that q * (1,0,0,0) * q* = (1, r, g, b)
 * Decode: q → extract rotated vector → RGB
 * =================================================================== */

/* Encode a color (r,g,b) ∈ [0,1] as a unit quaternion.
 * The color is mapped to a direction on the unit sphere.
 * q = normalize(1, 2r-1, 2g-1, 2b-1)
 *
 * This is lossless (within float precision) because the direction
 * is preserved by normalization and the amplitude carries brightness.
 */
void wubu_color_to_quat(float* q, float r, float g, float b) {
    q[0] = 1.0f;
    q[1] = r * 2.0f - 1.0f;
    q[2] = g * 2.0f - 1.0f;
    q[3] = b * 2.0f - 1.0f;
    wubu_quat_normalize(q, q);
}

/* Decode a unit quaternion back to color (r,g,b) ∈ [0,1].
 * Inverse of wubu_color_to_quat: r = (q.x/|q.w| + 1) / 2
 */
void wubu_quat_to_color(float* rgb, const float* q) {
    /* Since q = normalize(1, 2r-1, 2g-1, 2b-1), we have:
     * q.x / q.w = tanh-like mapping of r
     * For unit quaternion: q.w = 1/√(1 + (2r-1)² + (2g-1)² + (2b-1)²)
     * Recovery: r = (q.x / q.w + 1) / 2 */
    float inv_w = (fabsf(q[0]) > 1e-8f) ? (1.0f / q[0]) : 0.0f;
    rgb[0] = fminf(1.0f, fmaxf(0.0f, (q[1] * inv_w + 1.0f) * 0.5f));
    rgb[1] = fminf(1.0f, fmaxf(0.0f, (q[2] * inv_w + 1.0f) * 0.5f));
    rgb[2] = fminf(1.0f, fmaxf(0.0f, (q[3] * inv_w + 1.0f) * 0.5f));
}

/* Verify encode/decode is lossless */
float wubu_color_roundtrip_error(float r, float g, float b) {
    float q[4], rgb[3];
    wubu_color_to_quat(q, r, g, b);
    wubu_quat_to_color(rgb, q);
    float dr = fabsf(rgb[0] - r);
    float dg = fabsf(rgb[1] - g);
    float db = fabsf(rgb[2] - b);
    return fmaxf(dr, fmaxf(dg, db));
}
