/*
 * test_wubu_latent_codec.h
 */

#include <stdio.h>
#include <math.h>
#include "wubu_latent_codec.h"

static int pass = 0, fail = 0;

#define RUN_TEST(name) do { \
    printf("  %-50s ", #name "..."); fflush(stdout); \
    name(); printf("PASS\n"); pass++; \
} while(0)

#define ASSERT_TRUE(c) do { if(!(c)){printf("FAIL: %s\n",#c);fail++;return;} } while(0)

static void test_compress_decompress(void) {
    int N = 1000;
    float* q = (float*)malloc(N * 4 * sizeof(float));
    float* a = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N * 4; i++) q[i] = ((float)(i % 256) / 255.0f) * 2.0f - 1.0f;
    for (int i = 0; i < N; i++) a[i] = (float)(i % 256) / 255.0f;
    WubuCompressedLatent c = wubu_latent_compress(q, a, N, WUBU_QUALITY_HIGH);
    ASSERT_TRUE(c.compressed_size < c.raw_size);

    float* q_out = (float*)malloc(N * 4 * sizeof(float));
    float* a_out = (float*)malloc(N * sizeof(float));
    wubu_latent_decompress(q_out, a_out, &c);
    ASSERT_TRUE(fabsf(q_out[0] - q[0]) < 0.1f);
    wubu_latent_compressed_free(&c);
    free(q); free(a); free(q_out); free(a_out);
}

static void test_480p_compression(void) {
    int N = 854 * 480;
    float* q = (float*)malloc(N * 4 * sizeof(float));
    float* a = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N * 4; i++) q[i] = ((float)(i % 256) / 255.0f) * 2.0f - 1.0f;
    for (int i = 0; i < N; i++) a[i] = (float)(i % 256) / 255.0f;

    WubuCompressedLatent c = wubu_latent_compress(q, a, N, WUBU_QUALITY_MEDIUM);
    ASSERT_TRUE(c.compression_ratio > 2.0f);
    wubu_latent_print_stats(&c);
    wubu_latent_compressed_free(&c);
    free(q); free(a);
}

static void test_quality_levels(void) {
    float q[] = {0.2f, 0.5f, -0.1f, 0.8f};
    float a[] = {0.6f};
    WubuQuality qualities[] = {WUBU_QUALITY_LOSSLESS, WUBU_QUALITY_HIGH,
                                WUBU_QUALITY_MEDIUM, WUBU_QUALITY_LOW};
    for (int i = 0; i < 4; i++) {
        WubuCompressedLatent c = wubu_latent_compress(q, a, 1, qualities[i]);
        wubu_latent_compressed_free(&c);
    }
}

int main(void) {
    printf("=== WuBu Latent Codec Tests ===\n\n");
    RUN_TEST(test_compress_decompress);
    RUN_TEST(test_480p_compression);
    RUN_TEST(test_quality_levels);
    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
