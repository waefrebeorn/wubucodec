/*
 * test_wubu_tangent_flow.c -- Tests for tangent flow transformations
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_tangent_flow.h"

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
 * Linear flow: identity with scale
 * =================================================================== */

static void test_linear_flow_identity(void) {
    WubuTangentFlowConfig config = {
        .input_dim = 4,
        .hidden_dim = 8,
        .dropout = 0.0f,
        .flow_type = 1,
        .initial_scale = 1.0f
    };
    WubuTangentFlow flow;
    wubu_tangent_flow_init(&flow, &config);

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    wubu_tangent_flow_forward(&flow, x, out, 4);

    /* Linear flow with identity weights and scale=1: out = x */
    ASSERT_NEAR(wubu_tangent_flow_get_scale(&flow), 1.0f, 1e-3f);
    ASSERT_NEAR(out[0], 1.0f, 1e-3f);
    ASSERT_NEAR(out[1], 2.0f, 1e-3f);
    ASSERT_NEAR(out[2], 3.0f, 1e-3f);
    ASSERT_NEAR(out[3], 4.0f, 1e-3f);

    wubu_tangent_flow_free(&flow);
}

/* ===================================================================
 * MLP flow: output is finite
 * =================================================================== */

static void test_mlp_flow_finite(void) {
    WubuTangentFlowConfig config = {
        .input_dim = 8,
        .hidden_dim = 16,
        .dropout = 0.0f,
        .flow_type = 0,
        .initial_scale = 1.0f
    };
    WubuTangentFlow flow;
    wubu_tangent_flow_init(&flow, &config);

    float x[8];
    for (int i = 0; i < 8; i++) x[i] = (float)i * 0.1f;

    float out[8];
    wubu_tangent_flow_forward(&flow, x, out, 8);

    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(isfinite(out[i]));
    }

    wubu_tangent_flow_free(&flow);
}

/* ===================================================================
 * Scale constraint: scale > 0
 * =================================================================== */

static void test_scale_positive(void) {
    WubuTangentFlowConfig config = {
        .input_dim = 4,
        .hidden_dim = 8,
        .dropout = 0.0f,
        .flow_type = 0,
        .initial_scale = 0.5f
    };
    WubuTangentFlow flow;
    wubu_tangent_flow_init(&flow, &config);

    float scale = wubu_tangent_flow_get_scale(&flow);
    ASSERT_TRUE(scale > 0.0f);

    wubu_tangent_flow_free(&flow);
}

/* ===================================================================
 * Scale scaling: output scales with scale parameter
 * =================================================================== */

static void test_scale_scaling(void) {
    WubuTangentFlowConfig config1 = {
        .input_dim = 4, .hidden_dim = 8, .dropout = 0.0f,
        .flow_type = 1, .initial_scale = 1.0f
    };
    WubuTangentFlowConfig config2 = {
        .input_dim = 4, .hidden_dim = 8, .dropout = 0.0f,
        .flow_type = 1, .initial_scale = 2.0f
    };
    WubuTangentFlow flow1, flow2;
    wubu_tangent_flow_init(&flow1, &config1);
    wubu_tangent_flow_init(&flow2, &config2);

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out1[4], out2[4];
    wubu_tangent_flow_forward(&flow1, x, out1, 4);
    wubu_tangent_flow_forward(&flow2, x, out2, 4);

    /* out2 should be 2x out1 (since linear flow with scale=2 vs scale=1) */
    ASSERT_NEAR(out2[0], 2.0f * out1[0], 1e-3f);
    ASSERT_NEAR(out2[1], 2.0f * out1[1], 1e-3f);

    wubu_tangent_flow_free(&flow1);
    wubu_tangent_flow_free(&flow2);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuMath Tangent Flow Tests ===\n\n");

    RUN_TEST(test_linear_flow_identity);
    RUN_TEST(test_mlp_flow_finite);
    RUN_TEST(test_scale_positive);
    RUN_TEST(test_scale_scaling);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
