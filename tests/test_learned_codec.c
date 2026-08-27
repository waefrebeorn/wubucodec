/*
 * test_learned_codec.c — Test the learned codec inference pipeline
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_learned_codec.h"

static int pass = 0, fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fail++; } \
    else { pass++; } \
} while(0)

static void gen_test_image(float* rgb, int H, int W) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int i = (y * W + x) * 3;
            float u = (float)x / (float)(W - 1);
            float v = (float)y / (float)(H - 1);
            rgb[i] = u * 0.8f + 0.1f;
            rgb[i+1] = v * 0.6f + 0.2f;
            rgb[i+2] = (1.0f - u) * 0.5f + 0.2f;
        }
}

static void test_init_free(void) {
    printf("  Init/free...\n");
    WubuLearnedConfig cfg = wubu_learned_config_image(64, 64, 32, 8);
    WubuLearnedCodec codec;
    int ret = wubu_learned_init(&codec, &cfg);
    ASSERT(ret == 0, "init succeeds");
    wubu_learned_free(&codec);
}

static void test_encode_decode_pipeline(void) {
    printf("  Encode/decode pipeline...\n");
    int H = 64, W = 64;
    WubuLearnedConfig cfg = wubu_learned_config_image(H, W, 32, 8);
    WubuLearnedCodec codec;
    wubu_learned_init(&codec, &cfg);

    float* image = malloc(H * W * 3 * sizeof(float));
    float* latent = malloc(32 * sizeof(float));
    float* output = malloc(H * W * 3 * sizeof(float));

    gen_test_image(image, H, W);

    WubuEncodeState state;
    memset(&state, 0, sizeof(state));
    wubu_learned_encode(&codec, image, latent, &state);

    /* Check latent is finite */
    int finite = 1;
    for (int i = 0; i < 32; i++)
        if (!isfinite(latent[i])) { finite = 0; break; }
    ASSERT(finite, "latent is finite");

    /* Check latent is in expected range */
    int in_range = 1;
    for (int i = 0; i < 32; i++)
        if (latent[i] < -1.01f || latent[i] > 1.01f) { in_range = 0; break; }
    ASSERT(in_range, "latent in [-1,1] range");

    wubu_learned_decode(&codec, latent, &state, output);

    /* Check output is finite and in [0,1] */
    int out_ok = 1;
    for (int i = 0; i < H * W * 3; i++)
        if (!isfinite(output[i]) || output[i] < -0.01f || output[i] > 1.01f) { out_ok = 0; break; }
    ASSERT(out_ok, "output finite and in [0,1]");

    float psnr = wubu_learned_eval_psnr(&codec, image);
    printf("    PSNR: %.1f dB\n", psnr);
    ASSERT(isfinite(psnr), "PSNR is finite");
    ASSERT(psnr > 0.0f, "PSNR is positive");

    free(image); free(latent); free(output);
    wubu_learned_free(&codec);
}

static void test_compression_ratio(void) {
    printf("    Compression ratio...\n");

    /* 1024x1024 → 32 latent at 8-bit */
    WubuLearnedConfig cfg = wubu_learned_config_image(1024, 1024, 32, 8);
    WubuLearnedCodec codec;
    wubu_learned_init(&codec, &cfg);

    float ratio = wubu_learned_compression_ratio(&codec);
    printf("    1024x1024 → 32 latent @ 8-bit: %.0f:1\n", ratio);

    /* Raw: 1024*1024*3*4 = 12,582,912 bytes
       Compressed: 32 * 8/8 = 32 bytes
       Ratio: 12,582,912 / 32 = 393,216:1 */
    ASSERT(ratio > 1000.0f, "ratio > 1000:1 for 1024x1024→32@8bit");

    wubu_learned_free(&codec);
}

static void test_weight_save_load(void) {
    printf("    Weight save/load...\n");
    WubuLearnedConfig cfg = wubu_learned_config_image(64, 64, 16, 8);
    WubuLearnedCodec codec1, codec2;
    wubu_learned_init(&codec1, &cfg);
    wubu_learned_init(&codec2, &cfg);

    int H = 64, W = 64;
    float* image = malloc(H * W * 3 * sizeof(float));
    float* latent1 = malloc(16 * sizeof(float));
    float* latent2 = malloc(16 * sizeof(float));

    gen_test_image(image, H, W);

    /* Encode with codec1 */
    WubuEncodeState state1, state2;
    memset(&state1, 0, sizeof(state1));
    wubu_learned_encode(&codec1, image, latent1, &state1);

    /* Save and load weights */
    wubu_learned_save_weights(&codec1, "/home/wubu/.slermes/test_weights.bin");
    wubu_learned_load_weights(&codec2, "/home/wubu/.slermes/test_weights.bin");

    /* Encode with codec2 */
    memset(&state2, 0, sizeof(state2));
    wubu_learned_encode(&codec2, image, latent2, &state2);

    /* Check latents match */
    int match = 1;
    for (int i = 0; i < 16; i++)
        if (fabsf(latent1[i] - latent2[i]) > 1e-5f) { match = 0; break; }
    ASSERT(match, "latents match after save/load");

    free(image); free(latent1); free(latent2);
    wubu_learned_free(&codec1);
    wubu_learned_free(&codec2);
}

static void test_state_preserved(void) {
    printf("  State preserved across encode/decode...\n");
    int H = 64, W = 64;
    WubuLearnedConfig cfg = wubu_learned_config_image(H, W, 32, 8);
    WubuLearnedCodec codec;
    wubu_learned_init(&codec, &cfg);

    float* image = malloc(H * W * 3 * sizeof(float));
    float* latent = malloc(32 * sizeof(float));
    float* output = malloc(H * W * 3 * sizeof(float));

    gen_test_image(image, H, W);

    WubuEncodeState state;
    memset(&state, 0, sizeof(state));
    wubu_learned_encode(&codec, image, latent, &state);

    /* Verify state is populated */
    ASSERT(state.num_levels == 4, "state has 4 levels");
    ASSERT(state.Hp == 16, "state Hp correct (64/4)");
    ASSERT(state.Wp == 16, "state Wp correct (64/4)");
    ASSERT(state.patches != NULL, "state patches allocated");

    /* Verify encoder state values are finite */
    int finite = 1;
    for (int l = 0; l < 4; l++)
        for (int d = 0; d < state.level_dims[l]; d++)
            if (!isfinite(state.v_global[l][d])) { finite = 0; break; }
    ASSERT(finite, "encoder states are finite");

    wubu_learned_decode(&codec, latent, &state, output);

    /* Verify output differs from input (random weights won't reconstruct) */
    /* Just check that decode actually ran and produced non-trivial output */
    float out_mean = 0;
    for (int i = 0; i < H * W * 3; i++) out_mean += output[i];
    out_mean /= (float)(H * W * 3);
    ASSERT(out_mean > 0.0f && out_mean < 1.0f, "output mean in valid range");

    free(image); free(latent); free(output);
    wubu_learned_free(&codec);
}

int main(void) {
    printf("=== Learned Codec Tests ===\n\n");
    test_init_free();
    test_encode_decode_pipeline();
    test_compression_ratio();
    test_weight_save_load();
    test_state_preserved();
    printf("\n=== Results: %d pass, %d fail ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
