/*
 * wubu_so3.c -- SO(3) exp / log / geodesic.
 * Ported from tsotchke/libirrep/src/so3.c (MIT). Adapted to WuBuMath
 * Quat-style [w,x,y,z] is NOT used here (we work directly with Rot3 / axis-
 * angle / omega vectors); float precision per WuBuMath convention.
 *
 * Numerical guarantees verified in test_wubu_so3.c:
 *   - rot_exp(rot_log(R)) approx R  (round trip, < 1e-5)
 *   - rot_geodesic_distance(rot_exp(w), I) approx |w|  (< 1e-5)
 */

#include "wubu_so3.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* Below this rotation angle the full Rodrigues formula divides (1-cos) and
 * sin by theta, producing catastrophic cancellation; the O(theta) term
 * I + [w]_x is accurate to < 1e-6 at this threshold (float round-off). */
static const float ROT_EXP_SMALL = 1e-4f;

static inline float vec3_norm(const float v[3]) {
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Rot3 rot_identity(void) {
    return (Rot3){.m = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
}

Rot3 rot_compose(Rot3 A, Rot3 B) {
    Rot3 C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
                s += A.m[i * 3 + k] * B.m[k * 3 + j];
            C.m[i * 3 + j] = s;
        }
    return C;
}

Rot3 rot_inverse(Rot3 R) {
    /* SO(3) inverse is the transpose. */
    Rot3 T;
    T.m[0] = R.m[0]; T.m[1] = R.m[3]; T.m[2] = R.m[6];
    T.m[3] = R.m[1]; T.m[4] = R.m[4]; T.m[5] = R.m[7];
    T.m[6] = R.m[2]; T.m[7] = R.m[5]; T.m[8] = R.m[8];
    return T;
}

Rot3 rot_exp(const float omega[3]) {
    float theta = vec3_norm(omega);
    Rot3 R;
    if (theta < ROT_EXP_SMALL) {
        R.m[0] = 1; R.m[1] = -omega[2]; R.m[2] = omega[1];
        R.m[3] = omega[2]; R.m[4] = 1; R.m[5] = -omega[0];
        R.m[6] = -omega[1]; R.m[7] = omega[0]; R.m[8] = 1;
        return R;
    }
    float inv = 1.0f / theta;
    float nx = omega[0] * inv, ny = omega[1] * inv, nz = omega[2] * inv;
    float c = cosf(theta), s = sinf(theta), k = 1.0f - c;
    R.m[0] = c + nx * nx * k;
    R.m[1] = nx * ny * k - nz * s;
    R.m[2] = nx * nz * k + ny * s;
    R.m[3] = ny * nx * k + nz * s;
    R.m[4] = c + ny * ny * k;
    R.m[5] = ny * nz * k - nx * s;
    R.m[6] = nz * nx * k - ny * s;
    R.m[7] = nz * ny * k + nx * s;
    R.m[8] = c + nz * nz * k;
    return R;
}

/* Quaternion from rotation matrix (Shepperd 1978), w-last internally then
 * canonicalized. Returns [x,y,z,w]. */
static void quat_from_rot(Rot3 R, float q[4]) {
    float m00 = R.m[0], m01 = R.m[1], m02 = R.m[2];
    float m10 = R.m[3], m11 = R.m[4], m12 = R.m[5];
    float m20 = R.m[6], m21 = R.m[7], m22 = R.m[8];
    float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        q[3] = 0.25f * s;
        q[0] = (m21 - m12) / s;
        q[1] = (m02 - m20) / s;
        q[2] = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        float s = sqrtf(1.0f + m00 - m11 - m22) * 2.0f;
        q[3] = (m21 - m12) / s;
        q[0] = 0.25f * s;
        q[1] = (m01 + m10) / s;
        q[2] = (m02 + m20) / s;
    } else if (m11 > m22) {
        float s = sqrtf(1.0f + m11 - m00 - m22) * 2.0f;
        q[3] = (m02 - m20) / s;
        q[0] = (m01 + m10) / s;
        q[1] = 0.25f * s;
        q[2] = (m12 + m21) / s;
    } else {
        float s = sqrtf(1.0f + m22 - m00 - m11) * 2.0f;
        q[3] = (m10 - m01) / s;
        q[0] = (m02 + m20) / s;
        q[1] = (m12 + m21) / s;
        q[2] = 0.25f * s;
    }
    if (q[3] < 0.0f) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
}

/* Axis-angle from quaternion (canonical w>=0 so angle in [0,pi]). */
static AxisAngle axis_angle_from_quat(const float q[4]) {
    float vn = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    float angle = 2.0f * atan2f(vn, q[3]);
    if (vn < 1e-6f)
        return (AxisAngle){.axis = {0, 0, 1}, .angle = 0};
    float inv = 1.0f / vn;
    return (AxisAngle){.axis = {q[0] * inv, q[1] * inv, q[2] * inv}, .angle = angle};
}

void rot_log(Rot3 R, float omega_out[3]) {
    /* Route through quaternion (Shepperd) for pi-safety. */
    float q[4];
    quat_from_rot(R, q);
    AxisAngle aa = axis_angle_from_quat(q);
    omega_out[0] = aa.axis[0] * aa.angle;
    omega_out[1] = aa.axis[1] * aa.angle;
    omega_out[2] = aa.axis[2] * aa.angle;
}

AxisAngle rot_to_axis_angle(Rot3 R) {
    float q[4];
    quat_from_rot(R, q);
    return axis_angle_from_quat(q);
}

Rot3 rot_from_axis_angle(const float axis[3], float angle) {
    float an = vec3_norm(axis);
    if (an < 1e-8f)
        return rot_identity();
    float half = angle * 0.5f;
    float s = sinf(half) / an;
    float q[4] = {axis[0] * s, axis[1] * s, axis[2] * s, cosf(half)};
    /* quaternion -> matrix (w-last q[3]) */
    float xx = q[0]*q[0], yy = q[1]*q[1], zz = q[2]*q[2];
    float xy = q[0]*q[1], xz = q[0]*q[2], yz = q[1]*q[2];
    float wx = q[3]*q[0], wy = q[3]*q[1], wz = q[3]*q[2];
    Rot3 R;
    R.m[0] = 1 - 2*(yy + zz); R.m[1] = 2*(xy - wz);     R.m[2] = 2*(xz + wy);
    R.m[3] = 2*(xy + wz);     R.m[4] = 1 - 2*(xx + zz); R.m[5] = 2*(yz - wx);
    R.m[6] = 2*(xz - wy);     R.m[7] = 2*(yz + wx);     R.m[8] = 1 - 2*(xx + yy);
    return R;
}

float rot_geodesic_distance(Rot3 A, Rot3 B) {
    Rot3 Rd = rot_compose(rot_inverse(A), B);
    AxisAngle aa = rot_to_axis_angle(Rd);
    return aa.angle;
}
