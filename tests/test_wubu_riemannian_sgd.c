/*
 * test_wubu_riemannian_sgd.c -- Tests for Riemannian SGD optimizer
 *
 * Tests verify convergence behavior and numerical stability.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_riemannian_sgd.h"

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
 * Euclidean SGD: minimize f(x) = x^2
 * Gradient: 2x, update: x = x - lr * 2x
 * Should converge to 0
 * =================================================================== */

static void test_euclidean_sgd_convergence(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.1f,
        .initial_lr = 0.1f,
        .momentum_factor = 0.0f,
        .initial_momentum = 0.0f,
        .weight_decay = 0.0f,
        .max_grad_norm = 0.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 0.0f, .manifold_enabled = 0 };
    wubu_sgd_init(&opt, &config, manifold, 1);

    float x = 5.0f;
    for (int step = 0; step < 100; step++) {
        float grad = 2.0f * x;
        wubu_sgd_step_euclidean(&opt, &x, &grad, 1);
    }

    /* Should converge near 0 */
    ASSERT_TRUE(fabsf(x) < 0.01f);
    wubu_sgd_free(&opt);
}

/* ===================================================================
 * SGD with momentum: should converge faster
 * =================================================================== */

static void test_euclidean_sgd_momentum(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.1f,
        .initial_lr = 0.1f,
        .momentum_factor = 0.9f,
        .initial_momentum = 0.9f,
        .weight_decay = 0.0f,
        .max_grad_norm = 0.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 0.0f, .manifold_enabled = 0 };
    wubu_sgd_init(&opt, &config, manifold, 1);

    float x = 5.0f;
    for (int step = 0; step < 200; step++) {
        float grad = 2.0f * x;
        wubu_sgd_step_euclidean(&opt, &x, &grad, 1);
    }

    ASSERT_TRUE(fabsf(x) < 0.001f);
    wubu_sgd_free(&opt);
}

/* ===================================================================
 * Gradient clipping: large gradient should be clipped
 * =================================================================== */

static void test_gradient_clipping(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.1f,
        .initial_lr = 0.1f,
        .momentum_factor = 0.0f,
        .initial_momentum = 0.0f,
        .weight_decay = 0.0f,
        .max_grad_norm = 1.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 0.0f, .manifold_enabled = 0 };
    wubu_sgd_init(&opt, &config, manifold, 1);

    float x = 10.0f;
    float grad = 1000.0f; /* huge gradient */
    wubu_sgd_step_euclidean(&opt, &x, &grad, 1);

    /* With clipping to 1.0 and lr=0.1, step is at most 0.1 */
    ASSERT_TRUE(fabsf(x - 10.0f) <= 0.1f + 1e-5f);
    wubu_sgd_free(&opt);
}

/* ===================================================================
 * Hyperbolic SGD: parameter stays on manifold
 * =================================================================== */

static void test_hyperbolic_sgd_stays_on_manifold(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.01f,
        .initial_lr = 0.01f,
        .momentum_factor = 0.0f,
        .initial_momentum = 0.0f,
        .weight_decay = 0.0f,
        .max_grad_norm = 1.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 1.0f, .manifold_enabled = 1 };
    wubu_sgd_init(&opt, &config, manifold, 3);

    float param[] = {0.1f, 0.2f, 0.1f};
    float max_norm = 1.0f / sqrtf(1.0f);

    for (int step = 0; step < 50; step++) {
        float grad[] = {0.1f, -0.05f, 0.03f};
        wubu_sgd_step_hyperbolic(&opt, param, grad, 3);

        float norm = sqrtf(param[0]*param[0] + param[1]*param[1] + param[2]*param[2]);
        /* Must stay strictly inside the Poincare ball */
        if (norm >= max_norm) {
            printf("FAIL: parameter norm %g >= max %g at step %d\n", norm, max_norm, step);
            tests_failed++;
            wubu_sgd_free(&opt);
            return;
        }
    }

    wubu_sgd_free(&opt);
    tests_passed++;
    printf("  %-55s PASS\n", "test_hyperbolic_sgd_stays_on_manifold");
}

/* ===================================================================
 * Weight decay: should pull parameters toward zero
 * =================================================================== */

static void test_weight_decay(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.1f,
        .initial_lr = 0.1f,
        .momentum_factor = 0.0f,
        .initial_momentum = 0.0f,
        .weight_decay = 0.1f,
        .max_grad_norm = 0.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 0.0f, .manifold_enabled = 0 };
    wubu_sgd_init(&opt, &config, manifold, 1);

    float x = 1.0f;
    /* Zero gradient, but weight decay should still pull toward 0 */
    float grad = 0.0f;
    wubu_sgd_step_euclidean(&opt, &x, &grad, 1);

    ASSERT_TRUE(x < 1.0f); /* Should decrease */
    ASSERT_TRUE(x > 0.0f); /* Should not overshoot */
    wubu_sgd_free(&opt);
}

/* ===================================================================
 * LR setter: should clamp invalid values
 * =================================================================== */

static void test_lr_clamping(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.1f, .initial_lr = 0.1f,
        .momentum_factor = 0.0f, .initial_momentum = 0.0f,
        .weight_decay = 0.0f, .max_grad_norm = 0.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 0.0f, .manifold_enabled = 0 };
    wubu_sgd_init(&opt, &config, manifold, 1);

    wubu_sgd_set_lr(&opt, -1.0f);
    ASSERT_NEAR(opt.current_lr, 0.1f, 1e-6f); /* unchanged */

    wubu_sgd_set_lr(&opt, 0.5f);
    ASSERT_NEAR(opt.current_lr, 0.5f, 1e-6f); /* accepted */

    wubu_sgd_set_lr(&opt, 2.0f);
    ASSERT_NEAR(opt.current_lr, 0.5f, 1e-6f); /* rejected (>1) */

    wubu_sgd_free(&opt);
}

/* ===================================================================
 * Multi-dimensional Euclidean SGD
 * =================================================================== */

static void test_multidim_sgd(void) {
    WubuSGD opt;
    WubuSGDConfig config = {
        .learning_rate = 0.1f,
        .initial_lr = 0.1f,
        .momentum_factor = 0.9f,
        .initial_momentum = 0.9f,
        .weight_decay = 0.0f,
        .max_grad_norm = 0.0f,
        .q_controller_enabled = false
    };
    WubuManifoldBinding manifold = { .c = 0.0f, .manifold_enabled = 0 };
    wubu_sgd_init(&opt, &config, manifold, 3);

    float x[] = {3.0f, -2.0f, 1.0f};
    for (int step = 0; step < 200; step++) {
        float grad[] = {2.0f * x[0], 2.0f * x[1], 2.0f * x[2]};
        wubu_sgd_step_euclidean(&opt, x, grad, 3);
    }

    ASSERT_TRUE(fabsf(x[0]) < 0.001f);
    ASSERT_TRUE(fabsf(x[1]) < 0.001f);
    ASSERT_TRUE(fabsf(x[2]) < 0.001f);
    wubu_sgd_free(&opt);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuMath Riemannian SGD Tests ===\n\n");

    RUN_TEST(test_euclidean_sgd_convergence);
    RUN_TEST(test_euclidean_sgd_momentum);
    RUN_TEST(test_gradient_clipping);
    RUN_TEST(test_hyperbolic_sgd_stays_on_manifold);
    RUN_TEST(test_weight_decay);
    RUN_TEST(test_lr_clamping);
    RUN_TEST(test_multidim_sgd);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
