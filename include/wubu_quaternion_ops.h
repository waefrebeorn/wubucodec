/*
 * wubu_quaternion_ops.h -- Hamilton Quaternion Operations for WuBu
 *
 * Implements the proper WUBU math:
 *   - Hamilton product: p * q
 *   - Quaternion conjugate, normalize, inverse
 *   - Quaternion rotation: p * v * q
 *   - SLERP (spherical linear interpolation)
 *   - Poincaré ball exp/log maps, Möbius addition
 *   - WuBu nesting: inter-level tangent space rotation
 *   - Color↔quaternion encode/decode
 */

#ifndef WUBU_QUATERNION_OPS_H
#define WUBU_QUATERNION_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Basic Quaternion Operations
 * =================================================================== */

/* Hamilton product: p * q (both float[4]) */
void wubu_hamilton_product(float* out, const float* p, const float* q);

/* Batched Hamilton product for N quaternions [N, 4] */
void wubu_hamilton_product_batch(float* out, const float* p, const float* q, int N);

/* Quaternion conjugate: q* = (w, -x, -y, -z) */
void wubu_quat_conjugate(float* out, const float* q);

/* Quaternion norm squared */
float wubu_quat_norm_sq(const float* q);

/* Quaternion norm */
float wubu_quat_norm(const float* q);

/* Normalize quaternion to unit length */
void wubu_quat_normalize(float* out, const float* q);

/* Check if quaternion is unit length (|q| ≈ 1) */
int wubu_quat_is_unit(const float* q);

/* ===================================================================
 * Quaternion Rotation
 * =================================================================== */

/* Rotate vector v by unit quaternion p: R(v) = p * v * p* */
void wubu_quat_rotate_vector(float* out_vec, const float* p, const float* v);

/* Dual-quaternion rotation: R(v) = p * v * q */
void wubu_quat_rotate_dual(float* out_vec, const float* p, const float* q, const float* v);

/* Batched rotation */
void wubu_quat_rotate_batch(float* out, const float* p, const float* v, int N);

/* ===================================================================
 * SLERP: Spherical Linear Interpolation
 * =================================================================== */

/* Interpolate between two unit quaternions */
void wubu_quat_slerp(float* out, const float* p0, const float* p1, float t);

/* ===================================================================
 * Quaternion Exp/Log Maps
 * =================================================================== */

/* Quaternion exponential: exp(v) for pure quaternion v */
void wubu_quat_exp(float* out, const float* v);

/* Quaternion logarithm: log(q) for unit quaternion q */
void wubu_quat_log(float* out, const float* q);

/* ===================================================================
 * Poincaré Ball Operations (Hyperbolic Geometry)
 * =================================================================== */

/* Poincaré exponential map at origin */
void wubu_poincare_exp_map(float* out, const float* v, int D, float c);

/* Poincaré logarithmic map at origin */
void wubu_poincare_log_map(float* out, const float* y, int D, float c);

/* Möbius addition: defined in wubu_hyperbolic.c */

/* ===================================================================
 * WuBu Nesting: Inter-Level Tangent Space Rotation
 * =================================================================== */

/* Inter-level transition: log → rotate → exp */
void wubu_nesting_transition(float* out, const float* pos, int D,
                              const float* rot_p, const float* rot_q,
                              float c_current, float c_next);

/* ===================================================================
 * Color ↔ Quaternion Encode/Decode
 * =================================================================== */

/* Encode RGB color as unit quaternion */
void wubu_color_to_quat(float* q, float r, float g, float b);

/* Decode unit quaternion back to RGB color */
void wubu_quat_to_color(float* rgb, const float* q);

/* Verify encode/decode roundtrip error */
float wubu_color_roundtrip_error(float r, float g, float b);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_QUATERNION_OPS_H */
