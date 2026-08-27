/*
 * wubu_quaternion.c -- Quaternion operations for 3D rotation
 *
 * Slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py (bytropix/draftPY/)
 * Faithful C11 implementation of quaternion math
 */

#include "wubu_quaternion.h"
#include <math.h>

/* ===================================================================
 * Vector helpers
 * =================================================================== */

float vec3_dot(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

float vec3_norm(const float* v) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

void vec3_normalize(float* out, const float* v) {
    float n = vec3_norm(v);
    if (n < 1e-8f) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }
    float inv = 1.0f / n;
    out[0] = v[0] * inv;
    out[1] = v[1] * inv;
    out[2] = v[2] * inv;
}

void vec3_cross(float* out, const float* a, const float* b) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/* ===================================================================
 * Axis-Angle -> Quaternion
 * =================================================================== */

Quat quat_from_axis_angle(const float* axis, float angle) {
    float n[3];
    vec3_normalize(n, axis);

    float half_angle = angle * 0.5f;
    float s = sinf(half_angle);
    Quat q;
    q.w = cosf(half_angle);
    q.x = n[0] * s;
    q.y = n[1] * s;
    q.z = n[2] * s;
    return q;
}

/* ===================================================================
 * Quaternion Multiply (Hamilton product)
 * =================================================================== */

Quat quat_multiply(Quat q1, Quat q2) {
    Quat r;
    r.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
    r.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
    r.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
    r.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
    return r;
}

/* ===================================================================
 * Quaternion Conjugate
 * =================================================================== */

Quat quat_conjugate(Quat q) {
    Quat r;
    r.w = q.w;
    r.x = -q.x;
    r.y = -q.y;
    r.z = -q.z;
    return r;
}

/* ===================================================================
 * Rotate vector by quaternion: v' = q * v * q_conj
 * =================================================================== */

void quat_rotate_vector(float* out, Quat q, const float* v) {
    /* Treat v as pure quaternion (0, v[0], v[1], v[2]) */
    /* q * v_quat */
    float qv_w = -q.x*v[0] - q.y*v[1] - q.z*v[2];
    float qv_x =  q.w*v[0] + q.y*v[2] - q.z*v[1];
    float qv_y =  q.w*v[1] - q.x*v[2] + q.z*v[0];
    float qv_z =  q.w*v[2] + q.x*v[1] - q.y*v[0];

    /* (q * v_quat) * q_conj */
    float qconj_w = q.w, qconj_x = -q.x, qconj_y = -q.y, qconj_z = -q.z;
    out[0] = qv_w*qconj_x + qv_x*qconj_w + qv_y*qconj_z - qv_z*qconj_y;
    out[1] = qv_w*qconj_y - qv_x*qconj_z + qv_y*qconj_w + qv_z*qconj_x;
    out[2] = qv_w*qconj_z + qv_x*qconj_y - qv_y*qconj_x + qv_z*qconj_w;
}

/* ===================================================================
 * Normalize quaternion
 * =================================================================== */

Quat quat_normalize(Quat q) {
    float n = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (n < 1e-8f) {
        q.w = 1.0f; q.x = q.y = q.z = 0.0f;
        return q;
    }
    float inv = 1.0f / n;
    q.w *= inv; q.x *= inv; q.y *= inv; q.z *= inv;
    return q;
}

/* ===================================================================
 * Slerp (simplified -- uses normalized linear interpolation for small angles,
 *          falls back to full slerp for large angles)
 * =================================================================== */

Quat quat_slerp(Quat q1, Quat q2, float t) {
    float dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;

    /* Ensure shortest path */
    Quat q2_use = q2;
    if (dot < 0.0f) {
        q2_use.w = -q2.w; q2_use.x = -q2.x;
        q2_use.y = -q2.y; q2_use.z = -q2.z;
        dot = -dot;
    }

    Quat result;
    if (dot > 0.9995f) {
        /* Very close -- use normalized linear interpolation */
        result.w = q1.w + t * (q2_use.w - q1.w);
        result.x = q1.x + t * (q2_use.x - q1.x);
        result.y = q1.y + t * (q2_use.y - q1.y);
        result.z = q1.z + t * (q2_use.z - q1.z);
    } else {
        /* Full slerp */
        float theta_0 = acosf(dot);
        float theta = theta_0 * t;
        float sin_theta = sinf(theta);
        float sin_theta_0 = sinf(theta_0);

        float s1 = cosf(theta) - dot * sin_theta / sin_theta_0;
        float s2 = sin_theta / sin_theta_0;

        result.w = s1 * q1.w + s2 * q2_use.w;
        result.x = s1 * q1.x + s2 * q2_use.x;
        result.y = s1 * q1.y + s2 * q2_use.y;
        result.z = s1 * q1.z + s2 * q2_use.z;
    }

    return quat_normalize(result);
}
