/*
 * test_wubu_nest_gpt.c -- Tests for WuBuNestGPT model
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_nest_gpt.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name) do { \
    printf("  %-55s ", #name "..."); fflush(stdout); \
    name(); printf("PASS\n"); tests_passed++; \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", #cond); \
        tests_failed++; return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabsf((a)-(b)) > (tol)) { \
        printf("FAIL: %s=%g expected %g (tol=%g)\n", #a, (a), (b), (tol)); \
        tests_failed++; return; \
    } \
} while(0)

/* ===================================================================
 * Model init / free
 * =================================================================== */

static void test_embed_input_mode(void){
    /* GAP-D012: continuous features enter the SAME block stack; output
     * finite + deterministic across two calls. */
    WubuGPTConfig config = {
        .vocab_size = 64,.d_model = 64,.n_heads = 4,.d_head = 16,
        .d_compressed = 32,.d_ff = 128,.n_layers = 2,.dropout_rate = 0,
        .seed = 42 };
    WubuGPT m;
    ASSERT_TRUE(wubu_gpt_init(&m,&config)==0);
    const int B=1,T=8;
    float* feats=malloc(sizeof(float)*B*T*config.d_model);
    for(int i=0;i<B*T*config.d_model;i++)
        feats[i]=sinf(0.1f*i)*0.3f;
    float* h1=wubu_gpt_forward_embed(&m,feats,B,T,false);
    float* h2=wubu_gpt_forward_embed(&m,feats,B,T,false);
    ASSERT_TRUE(h1&&h2);
    for(int i=0;i<B*T*config.d_model;i++){
        ASSERT_TRUE(!isnan(h1[i])&&!isinf(h1[i]));
        ASSERT_TRUE(fabsf(h1[i]-h2[i])<1e-5f);
    }
    free(h1);free(h2);free(feats);
    wubu_gpt_free(&m);
}

static void test_gpt_init_free(void) {
    WubuGPTConfig config = {
        .vocab_size = 64,
        .d_model = 64,
        .n_heads = 4,
        .d_head = 16,
        .d_compressed = 32,
        .d_ff = 128,
        .n_layers = 2,
        .dropout_rate = 0,
        .seed = 42,
        .init_scale = 0.02f
    };
    WubuGPT model;
    int rc = wubu_gpt_init(&model, &config);
    ASSERT_TRUE(rc == 0);
    ASSERT_TRUE(model.D == 64);
    ASSERT_TRUE(model.H == 4);
    ASSERT_TRUE(model.N == 2);
    ASSERT_TRUE(model.V == 64);

    long params = wubu_gpt_count_params(&model);
    ASSERT_TRUE(params > 0);

    wubu_gpt_free(&model);
}

/* ===================================================================
 * Forward pass produces finite output
 * =================================================================== */

static void test_gpt_forward_finite(void) {
    WubuGPTConfig config = {
        .vocab_size = 64,
        .d_model = 64,
        .n_heads = 4,
        .d_head = 16,
        .d_compressed = 32,
        .d_ff = 128,
        .n_layers = 2,
        .dropout_rate = 0,
        .seed = 42,
        .init_scale = 0.02f
    };
    WubuGPT model;
    wubu_gpt_init(&model, &config);

    /* Small input */
    int tokens[] = {1, 5, 10, 20, 3, 7};
    int B = 2, T = 3;
    float* logits = wubu_gpt_forward(&model, tokens, B, T, false);
    ASSERT_TRUE(logits != NULL);

    /* Check all finite */
    for (int i = 0; i < B * T * model.V; i++) {
        ASSERT_TRUE(isfinite(logits[i]));
    }

    free(logits);
    wubu_gpt_free(&model);
}

/* ===================================================================
 * Loss computation
 * =================================================================== */

static void test_gpt_compute_loss(void) {
    WubuGPTConfig config = {
        .vocab_size = 32,
        .d_model = 32,
        .n_heads = 2,
        .d_head = 16,
        .d_compressed = 16,
        .d_ff = 64,
        .n_layers = 1,
        .dropout_rate = 0,
        .seed = 123,
        .init_scale = 0.02f
    };
    WubuGPT model;
    wubu_gpt_init(&model, &config);

    int B = 2, T = 2;
    int tokens[] = {1, 5, 10, 20};
    int targets[] = {5, 10, 20, 1};

    /* Forward to populate cache needed for loss */
    float* logits = wubu_gpt_forward(&model, tokens, B, T, false);
    ASSERT_TRUE(logits != NULL);
    free(logits);

    /* Compute loss via forward + manual cross-entropy would need cache access.
     * For now just verify model can do forward without crash. */
    wubu_gpt_free(&model);
}

/* ===================================================================
 * Parameter count matches formula
 * =================================================================== */

static void test_gpt_param_count(void) {
    WubuGPTConfig config = {
        .vocab_size = 16,
        .d_model = 32,
        .n_heads = 2,
        .d_head = 16,
        .d_compressed = 16,
        .d_ff = 64,
        .n_layers = 1,
        .dropout_rate = 0,
        .seed = 1,
        .init_scale = 0.02f
    };
    WubuGPT model;
    wubu_gpt_init(&model, &config);

    long params = wubu_gpt_count_params(&model);

    /* Manual count for 1 layer, D=32, V=16, H=2, d_h=16, d_c=16, d_ff=64 */
    long expected = 0;
    expected += 16 * 32;           /* wte */
    expected += 32 * 16;           /* lm_head */
    expected += 2 * 32;             /* final ln */
    expected += 4 * 32;             /* 2 layer norms * D */
    expected += 32 * 2 * 16;       /* wq */
    expected += 32 * 16;           /* wdkv */
    expected += 2 * 16 * 2 * 16;  /* wuk, wuv */
    expected += 2 * 32 * 2 * 8;   /* wqr, wkr (d_rope=8) */
    expected += 2 * 16 * 32;      /* wo */
    expected += 32 * 64 + 64 + 64 * 32 + 32; /* ffn */

    ASSERT_TRUE(params == expected);

    wubu_gpt_free(&model);
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(void) {
    printf("=== WuBuNestGPT Model Tests ===\n\n");

    RUN_TEST(test_gpt_init_free);
    RUN_TEST(test_gpt_forward_finite);
    RUN_TEST(test_gpt_compute_loss);
    RUN_TEST(test_gpt_param_count);
    RUN_TEST(test_embed_input_mode);

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
