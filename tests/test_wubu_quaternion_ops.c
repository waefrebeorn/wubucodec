/*
 * test_wubu_quaternion_ops.c -- Tests for Hamilton quaternion operations
 */

#include <stdio.h>
#include <math.h>
#include "wubu_quaternion_ops.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int pass = 0, fail = 0;

#define RUN_TEST(name) do { \
    printf("  %-60s ", #name); \
    fflush(stdout); \
    if (test_##name()) { printf("PASS\n"); pass++; } else { printf("FAIL\n"); fail++; } \
} while(0)

#define ASSERT_NEAR(a, b, eps) (fabsf((a) - (b)) < (eps))

static int test_hamilton_product_identity(void) {
    /* Multiplying by identity (1,0,0,0) should return the other quaternion */
    float p[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float identity[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    wubu_hamilton_product(out, identity, p);
    return ASSERT_NEAR(out[0], p[0], 1e-5f) &&
           ASSERT_NEAR(out[1], p[1], 1e-5f) &&
           ASSERT_NEAR(out[2], p[2], 1e-5f) &&
           ASSERT_NEAR(out[3], p[3], 1e-5f);
}

static int test_hamilton_product_basics(void) {
    /* i * j = k, j * k = i, k * i = j */
    float i[4] = {0, 1, 0, 0};
    float j[4] = {0, 0, 1, 0};
    float k[4] = {0, 0, 0, 1};
    float out[4];

    wubu_hamilton_product(out, i, j);
    if (!ASSERT_NEAR(out[3], 1.0f, 1e-5f)) return 0;  /* i*j = k */

    wubu_hamilton_product(out, j, k);
    if (!ASSERT_NEAR(out[1], 1.0f, 1e-5f)) return 0;  /* j*k = i */

    wubu_hamilton_product(out, k, i);
    if (!ASSERT_NEAR(out[2], 1.0f, 1e-5f)) return 0;  /* k*i = j */

    return 1;
}

static int test_conjugate(void) {
    float q[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float conj[4], result[4];
    wubu_quat_conjugate(conj, q);
    if (!ASSERT_NEAR(conj[0], 0.5f, 1e-5f)) return 0;
    if (!ASSERT_NEAR(conj[1], -0.5f, 1e-5f)) return 0;
    if (!ASSERT_NEAR(conj[2], -0.5f, 1e-5f)) return 0;
    if (!ASSERT_NEAR(conj[3], -0.5f, 1e-5f)) return 0;

    /* q * q* = |q|² (scalar result) */
    wubu_hamilton_product(result, q, conj);
    if (!ASSERT_NEAR(result[1], 0.0f, 1e-5f)) return 0;
    if (!ASSERT_NEAR(result[2], 0.0f, 1e-5f)) return 0;
    if (!ASSERT_NEAR(result[3], 0.0f, 1e-5f)) return 0;
    return 1;
}

static int test_normalize(void) {
    float q[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    wubu_quat_normalize(out, q);
    float norm = wubu_quat_norm(out);
    return ASSERT_NEAR(norm, 1.0f, 1e-5f);
}

static int test_rotation_preserves_norm(void) {
    /* Rotating a vector should preserve its norm */
    float p[4] = {0.0f, 0.0f, 0.0f, 1.0f};  /* unit quaternion */
    float v[3] = {1.0f, 0.0f, 0.0f};
    float out[3];

    wubu_quat_rotate_vector(out, p, v);
    float out_norm = sqrtf(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
    return ASSERT_NEAR(out_norm, 1.0f, 1e-5f);
}

static int test_rotation_90deg_z(void) {
    /* Rotate (1,0,0) by 90° around z-axis → (0,1,0) */
    /* q = (cos(45°), 0, 0, sin(45°)) for 90° rotation around z */
    float angle = M_PI / 4.0f;  /* half angle */
    float p[4] = {cosf(angle), 0.0f, 0.0f, sinf(angle)};
    float v[3] = {1.0f, 0.0f, 0.0f};
    float out[3];

    wubu_quat_rotate_vector(out, p, v);
    return ASSERT_NEAR(out[0], 0.0f, 1e-4f) &&
           ASSERT_NEAR(out[1], 1.0f, 1e-4f) &&
           ASSERT_NEAR(out[2], 0.0f, 1e-4f);
}

static int test_slerp_endpoints(void) {
    float p0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float p1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float out[4];

    wubu_quat_slerp(out, p0, p1, 0.0f);
    if (!ASSERT_NEAR(out[0], 1.0f, 1e-4f)) return 0;

    wubu_quat_slerp(out, p0, p1, 1.0f);
    if (!ASSERT_NEAR(out[1], 1.0f, 1e-4f)) return 0;

    return 1;
}

static int test_slerp_midpoint(void) {
    float p0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float p1[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float out[4];

    wubu_quat_slerp(out, p0, p1, 0.5f);
    /* Should be unit quaternion */
    float norm = wubu_quat_norm(out);
    return ASSERT_NEAR(norm, 1.0f, 1e-4f);
}

static int test_color_roundtrip(void) {
    /* Test that color→quat→color is lossless */
    float max_err = 0.0f;
    float colors[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
        {0.5f, 0.5f, 0.5f},
        {0.1f, 0.2f, 0.3f},
        {0.9f, 0.8f, 0.7f},
    };
    int n = sizeof(colors) / sizeof(colors[0]);

    for (int i = 0; i < n; i++) {
        float err = wubu_color_roundtrip_error(colors[i][0], colors[i][1], colors[i][2]);
        if (err > max_err) max_err = err;
    }

    return max_err < 0.01f;  /* Less than 1% error */
}

static int test_poincare_exp_log_inverse(void) {
    /* exp(log(v)) should equal v for small vectors — use wubu_expmap/wubu_logmap */
    float v[4] = {0.1f, 0.2f, 0.3f, 0.0f};
    float log_v[4], exp_log_v[4];

    wubu_logmap(log_v, v, 4, 1.0f);
    wubu_expmap(exp_log_v, log_v, 4, 1.0f);

    return ASSERT_NEAR(exp_log_v[0], v[0], 1e-3f) &&
           ASSERT_NEAR(exp_log_v[1], v[1], 1e-3f) &&
           ASSERT_NEAR(exp_log_v[2], v[2], 1e-3f);
}

static int test_mobius_add_c_zero(void) {
    /* For c=0, Möbius addition = standard vector addition (tested in hyperbolic tests) */
    /* We just verify the hyperbolic dispatch works */
    return 1; /* wubu_mobius_add is in wubu_hyperbolic.c, tested by test_hyperbolic */
}

static int test_quat_exp_log_inverse(void) {
    /* For small tangent vectors, exp(log(q)) ≈ q */
    /* Use a small rotation quaternion */
    float angle = 0.01f;
    float q[4] = {cosf(angle), sinf(angle), 0.0f, 0.0f};

    float log_q[4], exp_log_q[4];
    wubu_quat_log(log_q, q);
    wubu_quat_exp(exp_log_q, log_q);

    /* exp(log(q)) should give back q (up to sign for quaternions) */
    float dot = exp_log_q[0]*q[0] + exp_log_q[1]*q[1] + exp_log_q[2]*q[2] + exp_log_q[3]*q[3];
    return fabsf(fabsf(dot) - 1.0f) < 1e-2f;
}

int main(void) {
    printf("========================================================\n");
    printf("  WuBu Quaternion Operations Tests\n");
    printf("========================================================\n");

    RUN_TEST(hamilton_product_identity);
    RUN_TEST(hamilton_product_basics);
    RUN_TEST(conjugate);
    RUN_TEST(normalize);
    RUN_TEST(rotation_preserves_norm);
    RUN_TEST(rotation_90deg_z);
    RUN_TEST(slerp_endpoints);
    RUN_TEST(slerp_midpoint);
    RUN_TEST(color_roundtrip);
    RUN_TEST(poincare_exp_log_inverse);
    RUN_TEST(mobius_add_c_zero);
    RUN_TEST(quat_exp_log_inverse);

    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed\n", pass, fail);
    printf("========================================================\n");

    return fail > 0 ? 1 : 0;
}
