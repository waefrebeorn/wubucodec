/*
 * test_wubu_quaternion.c -- Tests for quaternion operations
 *
 * Tests derived from Python reference output (WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 */

#include <stdio.h>
#include <math.h>
#include "wubu_quaternion.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name) do { \
    printf("  %-55s ", #name "..."); fflush(stdout); \
    name(); printf("PASS\n"); tests_passed++; \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabsf((a)-(b)) > (tol)) { \
        printf("FAIL: %s=%g expected %g (tol=%g)\n", #a, (a), (b), (tol)); \
        tests_failed++; return; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", #cond); \
        tests_failed++; return; \
    } \
} while(0)

/* ===================================================================
 * Axis-angle tests
 * =================================================================== */

static void test_quat_from_axis_angle_zero(void) {
    float axis[] = {0.0f, 0.0f, 1.0f};
    Quat q = quat_from_axis_angle(axis, 0.0f);
    ASSERT_NEAR(q.w, 1.0f, 1e-6f);
    ASSERT_NEAR(q.x, 0.0f, 1e-6f);
    ASSERT_NEAR(q.y, 0.0f, 1e-6f);
    ASSERT_NEAR(q.z, 0.0f, 1e-6f);
}

static void test_quat_from_axis_angle_90z(void) {
    float axis[] = {0.0f, 0.0f, 1.0f};
    Quat q = quat_from_axis_angle(axis, M_PI / 2.0f);
    ASSERT_NEAR(q.w, cosf(M_PI / 4.0f), 1e-5f);
    ASSERT_NEAR(q.x, 0.0f, 1e-5f);
    ASSERT_NEAR(q.y, 0.0f, 1e-5f);
    ASSERT_NEAR(q.z, sinf(M_PI / 4.0f), 1e-5f);
}

static void test_quat_from_axis_angle_180x(void) {
    float axis[] = {1.0f, 0.0f, 0.0f};
    Quat q = quat_from_axis_angle(axis, M_PI);
    ASSERT_NEAR(q.w, 0.0f, 1e-5f);
    ASSERT_NEAR(q.x, 1.0f, 1e-5f);
    ASSERT_NEAR(q.y, 0.0f, 1e-5f);
    ASSERT_NEAR(q.z, 0.0f, 1e-5f);
}

/* ===================================================================
 * Multiply tests
 * =================================================================== */

static void test_quat_multiply_identity(void) {
    /* q * identity = q */
    Quat q = {0.5f, 0.5f, 0.5f, 0.5f};
    Quat identity = {1.0f, 0.0f, 0.0f, 0.0f};
    Quat r = quat_multiply(q, identity);
    ASSERT_NEAR(r.w, q.w, 1e-5f);
    ASSERT_NEAR(r.x, q.x, 1e-5f);
    ASSERT_NEAR(r.y, q.y, 1e-5f);
    ASSERT_NEAR(r.z, q.z, 1e-5f);
}

static void test_quat_multiply_conjugate(void) {
    /* q * q_conj = identity (for unit quaternion) */
    float axis[] = {1.0f, 1.0f, 0.0f};
    Quat q = quat_from_axis_angle(axis, M_PI / 3.0f);
    Quat q_conj = quat_conjugate(q);
    Quat r = quat_multiply(q, q_conj);
    ASSERT_NEAR(r.w, 1.0f, 1e-4f);
    ASSERT_NEAR(r.x, 0.0f, 1e-4f);
    ASSERT_NEAR(r.y, 0.0f, 1e-4f);
    ASSERT_NEAR(r.z, 0.0f, 1e-4f);
}

/* ===================================================================
 * Rotate vector tests
 * =================================================================== */

static void test_rotate_vector_zero_angle(void) {
    float axis[] = {0.0f, 0.0f, 1.0f};
    Quat q = quat_from_axis_angle(axis, 0.0f);
    float v[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    quat_rotate_vector(out, q, v);
    ASSERT_NEAR(out[0], 1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 2.0f, 1e-4f);
    ASSERT_NEAR(out[2], 3.0f, 1e-4f);
}

static void test_rotate_vector_180z(void) {
    /* 180 deg around Z: (x,y,z) -> (-x,-y,z) */
    float axis[] = {0.0f, 0.0f, 1.0f};
    Quat q = quat_from_axis_angle(axis, M_PI);
    float v[] = {1.0f, 0.0f, 0.0f};
    float out[3];
    quat_rotate_vector(out, q, v);
    ASSERT_NEAR(out[0], -1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 0.0f, 1e-4f);
    ASSERT_NEAR(out[2], 0.0f, 1e-4f);
}

static void test_rotate_vector_90z(void) {
    /* 90 deg around Z: (1,0,0) -> (0,1,0) */
    float axis[] = {0.0f, 0.0f, 1.0f};
    Quat q = quat_from_axis_angle(axis, M_PI / 2.0f);
    float v[] = {1.0f, 0.0f, 0.0f};
    float out[3];
    quat_rotate_vector(out, q, v);
    ASSERT_NEAR(out[0], 0.0f, 1e-4f);
    ASSERT_NEAR(out[1], 1.0f, 1e-4f);
    ASSERT_NEAR(out[2], 0.0f, 1e-4f);
}

static void test_rotate_vector_preserves_norm(void) {
    float axis[] = {1.0f, 1.0f, 1.0f};
    Quat q = quat_from_axis_angle(axis, M_PI / 4.0f);
    float v[] = {3.0f, -1.0f, 2.0f};
    float out[3];
    quat_rotate_vector(out, q, v);
    float v_norm = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    float o_norm = sqrtf(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
    ASSERT_NEAR(v_norm, o_norm, 1e-3f);
}

/* ===================================================================
 * Normalize tests
 * =================================================================== */

static void test_quat_normalize(void) {
    Quat q = {2.0f, 0.0f, 0.0f, 0.0f};
    Quat n = quat_normalize(q);
    ASSERT_NEAR(n.w, 1.0f, 1e-5f);
    ASSERT_NEAR(n.x, 0.0f, 1e-5f);
    ASSERT_NEAR(n.y, 0.0f, 1e-5f);
    ASSERT_NEAR(n.z, 0.0f, 1e-5f);
}

/* ===================================================================
 * Slerp tests
 * =================================================================== */

static void test_slerp_endpoints(void) {
    float axis[] = {0.0f, 1.0f, 0.0f};
    Quat q1 = quat_from_axis_angle(axis, 0.0f);
    Quat q2 = quat_from_axis_angle(axis, M_PI / 2.0f);

    Quat r0 = quat_slerp(q1, q2, 0.0f);
    ASSERT_NEAR(r0.w, q1.w, 1e-4f);

    Quat r1 = quat_slerp(q1, q2, 1.0f);
    ASSERT_NEAR(r1.w, q2.w, 1e-4f);
}

/* ===================================================================
 * Vector operations tests
 * =================================================================== */

static void test_vec3_cross(void) {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    float out[3];
    vec3_cross(out, a, b);
    ASSERT_NEAR(out[0], 0.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    ASSERT_NEAR(out[2], 1.0f, 1e-6f);
}

static void test_vec3_normalize(void) {
    float v[] = {3.0f, 4.0f, 0.0f};
    float out[3];
    vec3_normalize(out, v);
    ASSERT_NEAR(out[0], 0.6f, 1e-5f);
    ASSERT_NEAR(out[1], 0.8f, 1e-5f);
    ASSERT_NEAR(out[2], 0.0f, 1e-5f);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuMath Quaternion Tests ===\n\n");

    RUN_TEST(test_quat_from_axis_angle_zero);
    RUN_TEST(test_quat_from_axis_angle_90z);
    RUN_TEST(test_quat_from_axis_angle_180x);
    RUN_TEST(test_quat_multiply_identity);
    RUN_TEST(test_quat_multiply_conjugate);
    RUN_TEST(test_rotate_vector_zero_angle);
    RUN_TEST(test_rotate_vector_180z);
    RUN_TEST(test_rotate_vector_90z);
    RUN_TEST(test_rotate_vector_preserves_norm);
    RUN_TEST(test_quat_normalize);
    RUN_TEST(test_slerp_endpoints);
    RUN_TEST(test_vec3_cross);
    RUN_TEST(test_vec3_normalize);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
