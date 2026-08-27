/*
 * wubu_quaternion.h -- Quaternion operations for 3D rotation
 *
 * Slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py (bytropix/draftPY/)
 * Faithful C11 translation of quaternion_from_axis_angle, quaternion_multiply,
 * quaternion_apply_to_vector
 *
 * Features:
 *   - Quaternion from axis-angle representation
 *   - Quaternion multiplication (Hamilton product)
 *   - Rotate vector by quaternion (q * v * q_conj)
 *   - Quaternion conjugate and normalize
 *   - Quaternion slerp (spherical linear interpolation)
 */

#ifndef WUBU_QUATERNION_H
#define WUBU_QUATERNION_H

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Quaternion: [w, x, y, z] layout
 * =================================================================== */

typedef struct {
    float w, x, y, z;
} Quat;

/* ===================================================================
 * Axis-Angle -> Quaternion
 * ===================================================================
 * axis: [3] unit vector (will be normalized internally)
 * angle: rotation angle in radians
 * Output: Quat
 */

Quat quat_from_axis_angle(const float* axis, float angle);

/* ===================================================================
 * Quaternion Multiply (Hamilton product)
 * ===================================================================
 * q = q1 * q2
 */

Quat quat_multiply(Quat q1, Quat q2);

/* ===================================================================
 * Quaternion Conjugate
 * ===================================================================
 * conj(q) = [w, -x, -y, -z]
 */

Quat quat_conjugate(Quat q);

/* ===================================================================
 * Rotate vector by quaternion
 * ===================================================================
 * v_out = q * v * q_conj
 * Input:  q, v[3]
 * Output: v_out[3]
 */

void quat_rotate_vector(float* out, Quat q, const float* v);

/* ===================================================================
 * Normalize quaternion to unit length
 * =================================================================== */

Quat quat_normalize(Quat q);

/* ===================================================================
 * Spherical Linear Interpolation (Slerp)
 * ===================================================================
 * t: interpolation parameter [0, 1]
 * Output: normalized(q1 + t * (q2 - q1)) -- simplified for small angles
 */

Quat quat_slerp(Quat q1, Quat q2, float t);

/* ===================================================================
 * Vector operations
 * =================================================================== */

float vec3_dot(const float* a, const float* b);
float vec3_norm(const float* v);
void vec3_normalize(float* out, const float* v);
void vec3_cross(float* out, const float* a, const float* b);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_QUATERNION_H */
