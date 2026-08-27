/*
 * test_wubu_parallel_transport.c -- Tests for parallel transport
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_parallel_transport.h"

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
 * Euclidean parallel transport (identity)
 * =================================================================== */

static void test_pt_euclidean(void) {
    float v[] = {1.0f, 2.0f, 3.0f};
    float p[] = {0.5f, 0.5f, 0.5f};
    float out[3];
    wubu_parallel_transport_to_p(out, v, p, 3, 0.0f);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 2.0f, 1e-6f);
    ASSERT_NEAR(out[2], 3.0f, 1e-6f);
}

/* ===================================================================
 * Parallel transport to origin: identity at origin
 * =================================================================== */

static void test_pt_at_origin(void) {
    float v[] = {1.0f, 2.0f, 3.0f};
    float p[] = {0.0f, 0.0f, 0.0f};
    float out[3];
    wubu_parallel_transport_to_p(out, v, p, 3, 1.0f);
    ASSERT_NEAR(out[0], 1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 2.0f, 1e-4f);
    ASSERT_NEAR(out[2], 3.0f, 1e-4f);
}

/* ===================================================================
 * Lambda factor test
 * =================================================================== */

static void test_lambda_at_origin(void) {
    float p[] = {0.0f, 0.0f, 0.0f};
    float lam = wubu_lambda_factor(p, 3, 1.0f);
    ASSERT_NEAR(lam, 2.0f, 1e-4f);
}

/* ===================================================================
 * Lambda factor at radius
 * =================================================================== */

static void test_lambda_at_half_radius(void) {
    float p[] = {0.5f, 0.0f, 0.0f}; /* ||p|| = 0.5, c = 1 */
    float lam = wubu_lambda_factor(p, 3, 1.0f);
    float expected = 2.0f / (1.0f - 1.0f * 0.25f); /* 2 / 0.75 = 8/3 */
    ASSERT_NEAR(lam, expected, 1e-4f);
}

/* ===================================================================
 * Parallel transport round-trip: p->origin->p = identity
 * =================================================================== */

static void test_pt_roundtrip_to_p(void) {
    float v[] = {0.5f, -0.3f, 0.1f};
    float p[] = {0.2f, 0.1f, -0.3f};
    float v_at_p[3], v_back[3];
    wubu_parallel_transport_to_origin(v_at_p, v, p, 3, 1.0f);
    wubu_parallel_transport_to_p(v_back, v_at_p, p, 3, 1.0f);
    ASSERT_NEAR(v_back[0], v[0], 1e-4f);
    ASSERT_NEAR(v_back[1], v[1], 1e-4f);
    ASSERT_NEAR(v_back[2], v[2], 1e-4f);
}

/* ===================================================================
 * Parallel transport round-trip: origin->p->origin = identity
 * =================================================================== */

static void test_pt_roundtrip_to_origin(void) {
    float v[] = {0.5f, -0.3f, 0.1f};
    float p[] = {0.2f, 0.1f, -0.3f};
    float v_at_p[3], v_back[3];
    wubu_parallel_transport_to_p(v_at_p, v, p, 3, 1.0f);
    wubu_parallel_transport_to_origin(v_back, v_at_p, p, 3, 1.0f);
    ASSERT_NEAR(v_back[0], v[0], 1e-4f);
    ASSERT_NEAR(v_back[1], v[1], 1e-4f);
    ASSERT_NEAR(v_back[2], v[2], 1e-4f);
}

/* ===================================================================
 * Parallel transport: p->q via origin
 * =================================================================== */

static void test_pt_via_origin(void) {
    float v[] = {1.0f, 0.0f, 0.0f};
    float p[] = {0.1f, 0.0f, 0.0f};
    float q[] = {0.0f, 0.1f, 0.0f};
    float out[3];
    wubu_parallel_transport(out, v, p, q, 3, 1.0f);
    /* Should produce a finite result */
    ASSERT_TRUE(isfinite(out[0]));
    ASSERT_TRUE(isfinite(out[1]));
    ASSERT_TRUE(isfinite(out[2]));
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuMath Parallel Transport Tests ===\n\n");

    RUN_TEST(test_pt_euclidean);
    RUN_TEST(test_pt_at_origin);
    RUN_TEST(test_lambda_at_origin);
    RUN_TEST(test_lambda_at_half_radius);
    RUN_TEST(test_pt_roundtrip_to_p);
    RUN_TEST(test_pt_roundtrip_to_origin);
    RUN_TEST(test_pt_via_origin);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
