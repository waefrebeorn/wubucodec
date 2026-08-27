/*
 * test_wubu_hyperbolic.c -- Tests for Poincare ball hyperbolic geometry
 *
 * Tests derived from Python reference output (WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * Values verified by running the Python implementation and comparing.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_hyperbolic.h"

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
 * Poincare Clip tests
 * =================================================================== */

static void test_poincare_clip_euclidean(void) {
    /* c <= 0: should just copy */
    float x[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    wubu_poincare_clip(out, x, 3, 0.0f);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 2.0f, 1e-6f);
    ASSERT_NEAR(out[2], 3.0f, 1e-6f);
}

static void test_poincare_clip_inside(void) {
    /* c=1, small vector: should be unchanged */
    float x[] = {0.1f, 0.2f, 0.1f};
    float out[3];
    wubu_poincare_clip(out, x, 3, 1.0f);
    ASSERT_NEAR(out[0], 0.1f, 1e-4f);
    ASSERT_NEAR(out[1], 0.2f, 1e-4f);
    ASSERT_NEAR(out[2], 0.1f, 1e-4f);
}

static void test_poincare_clip_outside(void) {
    /* c=1, large vector: should be scaled down */
    float x[] = {10.0f, 10.0f, 10.0f};
    float out[3];
    wubu_poincare_clip(out, x, 3, 1.0f);
    float norm = sqrtf(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
    float max_norm = 1.0f / sqrtf(1.0f + WUBU_EPS);
    max_norm *= (1.0f - WUBU_EPS * 10.0f);
    ASSERT_TRUE(norm < max_norm + 1e-4f);
}

/* ===================================================================
 * Exponential map tests
 * =================================================================== */

static void test_expmap_euclidean(void) {
    /* c <= 0: identity */
    float v[] = {0.5f, 0.3f, 0.1f};
    float out[3];
    wubu_expmap(out, v, 3, 0.0f);
    ASSERT_NEAR(out[0], 0.5f, 1e-6f);
    ASSERT_NEAR(out[1], 0.3f, 1e-6f);
    ASSERT_NEAR(out[2], 0.1f, 1e-6f);
}

static void test_expmap_near_origin(void) {
    /* Near origin: expmap_0(v) ≈ v */
    float v[] = {1e-8f, 1e-8f, 1e-8f};
    float out[3];
    wubu_expmap(out, v, 3, 1.0f);
    ASSERT_NEAR(out[0], 1e-8f, 1e-6f);
    ASSERT_NEAR(out[1], 1e-8f, 1e-6f);
    ASSERT_NEAR(out[2], 1e-8f, 1e-6f);
}

static void test_expmap_result_inside(void) {
    /* Result must be inside Poincare ball */
    float v[] = {1.0f, 1.0f, 1.0f};
    float out[3];
    wubu_expmap(out, v, 3, 1.0f);
    float norm = sqrtf(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
    ASSERT_TRUE(norm < 1.0f / sqrtf(1.0f) + 1e-3f);
}

/* ===================================================================
 * Logarithmic map tests
 * =================================================================== */

static void test_logmap_euclidean(void) {
    /* c <= 0: identity */
    float y[] = {0.5f, 0.3f, 0.1f};
    float out[3];
    wubu_logmap(out, y, 3, 0.0f);
    ASSERT_NEAR(out[0], 0.5f, 1e-6f);
    ASSERT_NEAR(out[1], 0.3f, 1e-6f);
    ASSERT_NEAR(out[2], 0.1f, 1e-6f);
}

static void test_logmap_near_origin(void) {
    /* Near origin: logmap_0(y) ≈ y */
    float y[] = {1e-8f, 1e-8f, 1e-8f};
    float out[3];
    wubu_logmap(out, y, 3, 1.0f);
    ASSERT_NEAR(out[0], 1e-8f, 1e-6f);
    ASSERT_NEAR(out[1], 1e-8f, 1e-6f);
    ASSERT_NEAR(out[2], 1e-8f, 1e-6f);
}

/* ===================================================================
 * Round-trip: logmap(expmap(v)) ≈ v
 * =================================================================== */

static void test_roundtrip_expmap_logmap(void) {
    float v[] = {0.3f, -0.1f, 0.5f};
    float exp_out[3], log_out[3];
    wubu_expmap(exp_out, v, 3, 1.0f);
    wubu_logmap(log_out, exp_out, 3, 1.0f);
    ASSERT_NEAR(log_out[0], v[0], 1e-3f);
    ASSERT_NEAR(log_out[1], v[1], 1e-3f);
    ASSERT_NEAR(log_out[2], v[2], 1e-3f);
}

/* ===================================================================
 * Mobius addition tests
 * =================================================================== */

static void test_mobius_add_euclidean(void) {
    /* c <= 0: standard addition */
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {0.5f, 0.5f, 0.5f};
    float out[3];
    wubu_mobius_add(out, x, y, 3, 0.0f);
    ASSERT_NEAR(out[0], 1.5f, 1e-6f);
    ASSERT_NEAR(out[1], 2.5f, 1e-6f);
    ASSERT_NEAR(out[2], 3.5f, 1e-6f);
}

static void test_mobius_add_identity(void) {
    /* x (+) 0 = x */
    float x[] = {0.3f, 0.2f, 0.1f};
    float zero[] = {0.0f, 0.0f, 0.0f};
    float out[3];
    wubu_mobius_add(out, x, zero, 3, 1.0f);
    ASSERT_NEAR(out[0], 0.3f, 1e-4f);
    ASSERT_NEAR(out[1], 0.2f, 1e-4f);
    ASSERT_NEAR(out[2], 0.1f, 1e-4f);
}

/* ===================================================================
 * egrad2rgrad tests
 * =================================================================== */

static void test_egrad2rgrad_euclidean(void) {
    /* c <= 0: identity */
    float p[] = {0.5f, 0.5f, 0.5f};
    float g[] = {1.0f, 0.0f, -1.0f};
    float out[3];
    wubu_egrad2rgrad(out, p, g, 3, 0.0f);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[1], 0.0f, 1e-6f);
    ASSERT_NEAR(out[2], -1.0f, 1e-6f);
}

static void test_egrad2rgrad_at_origin(void) {
    /* At origin: scaling factor = ((1 - c*0)/2)^2 = 0.25 */
    float p[] = {0.0f, 0.0f, 0.0f};
    float g[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    wubu_egrad2rgrad(out, p, g, 3, 1.0f);
    ASSERT_NEAR(out[0], 0.25f, 1e-4f);
    ASSERT_NEAR(out[1], 0.50f, 1e-4f);
    ASSERT_NEAR(out[2], 0.75f, 1e-4f);
}

/* ===================================================================
 * Hyperbolic distance tests
 * =================================================================== */

static void test_hyperbolic_distance_euclidean(void) {
    /* c <= 0: Euclidean distance */
    float x[] = {0.0f, 0.0f, 0.0f};
    float y[] = {3.0f, 4.0f, 0.0f};
    float d = wubu_hyperbolic_distance(x, y, 3, 0.0f);
    ASSERT_NEAR(d, 5.0f, 1e-6f);
}

static void test_hyperbolic_distance_self(void) {
    /* d(x, x) = 0 */
    float x[] = {0.3f, 0.2f, 0.1f};
    float d = wubu_hyperbolic_distance(x, x, 3, 1.0f);
    ASSERT_NEAR(d, 0.0f, 1e-4f);
}

/* ===================================================================
 * Scale-aware tests
 * =================================================================== */

static void test_expmap_scaled_euclidean(void) {
    float v[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    wubu_expmap_scaled(out, v, 3, 0.0f, 2.0f);
    ASSERT_NEAR(out[0], 2.0f, 1e-6f);
    ASSERT_NEAR(out[1], 4.0f, 1e-6f);
    ASSERT_NEAR(out[2], 6.0f, 1e-6f);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuMath Hyperbolic Geometry Tests ===\n\n");

    RUN_TEST(test_poincare_clip_euclidean);
    RUN_TEST(test_poincare_clip_inside);
    RUN_TEST(test_poincare_clip_outside);
    RUN_TEST(test_expmap_euclidean);
    RUN_TEST(test_expmap_near_origin);
    RUN_TEST(test_expmap_result_inside);
    RUN_TEST(test_logmap_euclidean);
    RUN_TEST(test_logmap_near_origin);
    RUN_TEST(test_roundtrip_expmap_logmap);
    RUN_TEST(test_mobius_add_euclidean);
    RUN_TEST(test_mobius_add_identity);
    RUN_TEST(test_egrad2rgrad_euclidean);
    RUN_TEST(test_egrad2rgrad_at_origin);
    RUN_TEST(test_hyperbolic_distance_euclidean);
    RUN_TEST(test_hyperbolic_distance_self);
    RUN_TEST(test_expmap_scaled_euclidean);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
