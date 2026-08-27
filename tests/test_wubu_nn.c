/*
 * test_wubu_nn.c -- Unit tests for the C11 Neural Network Layer Library
 */

#include "wubu_nn.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  test_%s... ", #name)
#define PASS() printf("PASS\n")
#define FAIL(msg) printf("FAIL: %s\n", msg)

static int test_count = 0;
static int pass_count = 0;

static void check_close(const float* a, const float* b, int n, const char* name) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    test_count++;
    if (max_err < 1e-4f) {
        pass_count++;
        PASS();
    } else {
        FAIL(name);
        printf("    max_err=%.6e\n", max_err);
    }
}

static void check_scalar(float actual, float expected, const char* name) {
    test_count++;
    if (fabsf(actual - expected) < 1e-3f) {
        pass_count++;
        PASS();
    } else {
        FAIL(name);
        printf("    actual=%.6f expected=%.6f\n", actual, expected);
    }
}

static void test_linear(void) {
    TEST(linear);
    float weight[12] = {0.1f,0.2f,0.3f,0.4f, 0.5f,0.6f,0.7f,0.8f, 0.9f,1.0f,1.1f,1.2f};
    float bias[3] = {0.01f, 0.02f, 0.03f};
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[3] = {0};
    WubuLinear layer;
    layer.in_features = 4; layer.out_features = 3;
    layer.weight = weight; layer.bias = bias;
    wubu_linear_forward(&layer, x, output, 1);
    float expected[3] = {3.01f, 7.02f, 11.03f};
    check_close(output, expected, 3, "linear_output");
}

static void test_gelu(void) {
    TEST(gelu);
    float x[4] = {1.0f, 0.0f, -1.0f, 2.0f};
    float output[4] = {0};
    wubu_gelu_forward(x, output, 4);
    float expected[4];
    expected[0] = 1.0f * 0.5f * (1.0f + erff(1.0f / 1.41421356237f));
    expected[1] = 0.0f;
    expected[2] = -1.0f * 0.5f * (1.0f + erff(-1.0f / 1.41421356237f));
    expected[3] = 2.0f * 0.5f * (1.0f + erff(2.0f / 1.41421356237f));
    check_close(output, expected, 4, "gelu_output");
}

static void test_layer_norm(void) {
    TEST(layer_norm);
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    WubuLayerNorm layer;
    layer.normalized_shape = 4; layer.eps = 1e-5f;
    layer.weight = (float*)malloc(4 * sizeof(float));
    layer.bias = (float*)malloc(4 * sizeof(float));
    wubu_init_ones(layer.weight, 4);
    wubu_init_zeros(layer.bias, 4);
    wubu_layer_norm_forward(&layer, x, output, 1);
    float mean = 2.5f, var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    float expected[4];
    for (int i = 0; i < 4; ++i) expected[i] = (x[i] - mean) * inv_std;
    check_close(output, expected, 4, "layer_norm_output");
    free(layer.weight); free(layer.bias);
}

static void test_softmax(void) {
    TEST(softmax);
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    wubu_softmax(x, output, 4);
    float mx = 4.0f, sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += expf(x[i] - mx);
    float expected[4];
    for (int i = 0; i < 4; ++i) expected[i] = expf(x[i] - mx) / sum;
    check_close(output, expected, 4, "softmax_output");
}

static void test_cross_entropy(void) {
    TEST(cross_entropy);
    float logits[6] = {2.0f, 1.0f, 0.1f, 0.5f, 2.5f, 0.3f};
    int targets[2] = {0, 1};
    float loss = wubu_cross_entropy_loss(logits, targets, 2, 3);
    check_scalar(loss, 0.318f, "cross_entropy_value");
}

static void test_mse(void) {
    TEST(mse);
    float pred[3] = {1.0f, 2.0f, 3.0f};
    float target[3] = {1.5f, 2.5f, 3.5f};
    float loss = wubu_mse_loss(pred, target, 3);
    check_scalar(loss, 0.25f, "mse_value");
}

static void test_bce(void) {
    TEST(bce_with_logits);
    float pred[3] = {0.5f, -0.5f, 1.0f};
    float target[3] = {1.0f, 0.0f, 1.0f};
    float loss = wubu_bce_with_logits_loss(pred, target, 3);
    check_scalar(loss, 0.420f, "bce_value");
}

static void test_clip_grad(void) {
    TEST(clip_grad_norm);
    float grads[2] = {3.0f, 4.0f};
    wubu_clip_grad_norm(grads, 2, 2.5f);
    float expected[2] = {1.5f, 2.0f};
    check_close(grads, expected, 2, "clipped_grads");
}

static void test_embedding(void) {
    TEST(embedding);
    WubuEmbedding emb;
    wubu_embedding_init(&emb, 5, 3);
    for (int i = 0; i < 15; ++i) emb.weight[i] = (float)i * 0.1f;
    int indices[3] = {2, 0, 4};
    float output[9] = {0};
    wubu_embedding_forward(&emb, indices, output, 3);
    float expected[9] = {0.6f,0.7f,0.8f, 0.0f,0.1f,0.2f, 1.2f,1.3f,1.4f};
    check_close(output, expected, 9, "embedding_output");
    wubu_embedding_free(&emb);
}

static void test_adaptive_pool(void) {
    TEST(adaptive_avg_pool);
    float x[16];
    for (int i = 0; i < 16; ++i) x[i] = (float)(i + 1);
    float output[4] = {0};
    wubu_adaptive_avg_pool2d(x, 1, 1, 4, 4, 2, 2, output);
    float expected[4] = {3.5f, 5.5f, 11.5f, 13.5f};
    check_close(output, expected, 4, "adaptive_pool_output");
}

static void test_group_norm(void) {
    TEST(group_norm);
    int B = 1, C = 4, H = 2, W_img = 2;
    float x[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    float output[16] = {0};
    WubuGroupNorm gn;
    wubu_group_norm_init(&gn, 2, C, 1e-5f);
    wubu_group_norm_forward(&gn, x, B, C, H, W_img, output);
    test_count++;
    int finite = 1;
    for (int i = 0; i < B*C*H*W_img; ++i) {
        if (!isfinite(output[i])) { finite = 0; break; }
    }
    if (finite) { pass_count++; PASS(); }
    else { FAIL("group_norm produced non-finite values"); }
    wubu_group_norm_free(&gn);
}

static void test_conv2d(void) {
    TEST(conv2d);
    float x[9] = {1,2,3, 4,5,6, 7,8,9};
    float output[9] = {0};
    WubuConv2d conv;
    wubu_conv2d_init(&conv, 1, 1, 3, 1, 1, 0);
    /* Set identity kernel: center=1, rest=0 */
    memset(conv.weight, 0, 9 * sizeof(float));
    conv.weight[4] = 1.0f;  /* center of 3x3 kernel */
    wubu_conv2d_forward(&conv, x, 1, 3, 3, output);
    test_count++;
    if (fabsf(output[4] - 5.0f) < 1e-5f) { pass_count++; PASS(); }
    else { FAIL("conv2d center pixel wrong"); printf("    output[4]=%f\n", output[4]); }
    wubu_conv2d_free(&conv);
}

static void test_dropout(void) {
    TEST(dropout);
    float x[1000];
    for (int i = 0; i < 1000; ++i) x[i] = 1.0f;
    float output[1000];
    WubuRNG rng;
    wubu_rng_init(&rng, 42);
    wubu_dropout_forward(x, output, 1000, 0.5f, &rng);
    int zeros = 0;
    float sum = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        if (output[i] == 0.0f) zeros++;
        sum += output[i];
    }
    test_count++;
    if (zeros > 400 && zeros < 600 && sum > 900.0f && sum < 1100.0f) { pass_count++; PASS(); }
    else { FAIL("dropout statistics out of range"); printf("    zeros=%d sum=%f\n", zeros, sum); }
}

static void test_silu(void) {
    TEST(silu);
    float x[3] = {0.0f, 1.0f, -1.0f};
    float output[3] = {0};
    wubu_silu_forward(x, output, 3);
    float expected[3];
    expected[0] = 0.0f;
    expected[1] = 1.0f * sigmoid(1.0f);
    expected[2] = -1.0f * sigmoid(-1.0f);
    check_close(output, expected, 3, "silu_output");
}

static void test_tanh(void) {
    TEST(tanh);
    float x[3] = {0.0f, 0.5f, -0.5f};
    float output[3] = {0};
    wubu_tanh_forward(x, output, 3);
    float expected[3] = {tanhf(0.0f), tanhf(0.5f), tanhf(-0.5f)};
    check_close(output, expected, 3, "tanh_output");
}

static void test_leaky_relu(void) {
    TEST(leaky_relu);
    float x[4] = {1.0f, -1.0f, 0.0f, -0.5f};
    float output[4] = {0};
    wubu_leaky_relu_forward(x, output, 4, 0.01f);
    float expected[4] = {1.0f, -0.01f, 0.0f, -0.005f};
    check_close(output, expected, 4, "leaky_relu_output");
}

static void test_relu(void) {
    TEST(relu);
    float x[4] = {1.0f, -1.0f, 0.0f, 0.5f};
    float output[4] = {0};
    wubu_relu_forward(x, output, 4);
    float expected[4] = {1.0f, 0.0f, 0.0f, 0.5f};
    check_close(output, expected, 4, "relu_output");
}

static void test_xavier_init(void) {
    TEST(xavier_uniform);
    float weights[10000];
    wubu_init_xavier_uniform(weights, 10000, 100, 50);
    float limit = sqrtf(6.0f / 150.0f);
    int in_range = 1;
    float sum = 0.0f;
    for (int i = 0; i < 10000; ++i) {
        if (weights[i] < -limit || weights[i] > limit) { in_range = 0; break; }
        sum += weights[i];
    }
    test_count++;
    if (in_range && fabsf(sum) < 100.0f) { pass_count++; PASS(); }
    else { FAIL("xavier init out of range or biased"); }
}

int main(void) {
    printf("=== WuBuMath NN Layer Tests ===\n\n");
    test_linear();
    test_gelu();
    test_layer_norm();
    test_softmax();
    test_cross_entropy();
    test_mse();
    test_bce();
    test_clip_grad();
    test_embedding();
    test_adaptive_pool();
    test_group_norm();
    test_conv2d();
    test_dropout();
    test_silu();
    test_tanh();
    test_leaky_relu();
    test_relu();
    test_xavier_init();
    printf("\n=== Results ===\n");
    printf("Passed: %d\n", pass_count);
    printf("Failed: %d\n", test_count - pass_count);
    printf("Total:  %d\n", test_count);
    return (pass_count == test_count) ? 0 : 1;
}
