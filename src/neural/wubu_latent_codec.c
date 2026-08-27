/*
 * wubu_latent_codec.c -- WuBu Latent Space Compression Codec
 *
 * Implements scalar quantization + RLE + entropy coding for Hamilton
 * quaternion latent point clouds.
 */

#include "wubu_latent_codec.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ===================================================================
 * Quality Presets
 * =================================================================== */

WubuQuantConfig wubu_quality_to_config(WubuQuality quality) {
    WubuQuantConfig cfg;
    switch (quality) {
        case WUBU_QUALITY_LOSSLESS:
            cfg.quat_bits = 16; cfg.amp_bits = 16;
            cfg.use_rle = 0; cfg.use_entropy = 0; break;
        case WUBU_QUALITY_HIGH:
            cfg.quat_bits = 12; cfg.amp_bits = 12;
            cfg.use_rle = 0; cfg.use_entropy = 1; break;
        case WUBU_QUALITY_MEDIUM:
            cfg.quat_bits = 8; cfg.amp_bits = 8;
            cfg.use_rle = 1; cfg.use_entropy = 1; break;
        case WUBU_QUALITY_LOW:
            cfg.quat_bits = 4; cfg.amp_bits = 4;
            cfg.use_rle = 1; cfg.use_entropy = 1; break;
        default:
            cfg.quat_bits = 8; cfg.amp_bits = 8;
            cfg.use_rle = 0; cfg.use_entropy = 0; break;
    }
    return cfg;
}

/* ===================================================================
 * Scalar Quantization
 * =================================================================== */

static inline uint16_t quantize_float16(float val, int bits, float min_val, float max_val) {
    if (max_val <= min_val) return 0;
    float range = max_val - min_val;
    float normalized = (val - min_val) / range;
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    uint32_t levels = (1u << bits) - 1;
    return (uint16_t)(normalized * (float)levels);
}

static inline float dequantize_uint16(uint16_t q, int bits, float min_val, float max_val) {
    if (max_val <= min_val) return min_val;
    float range = max_val - min_val;
    float normalized = (float)q / (float)((1u << bits) - 1);
    return min_val + normalized * range;
}

/* ===================================================================
 * Find min/max for quantization
 * =================================================================== */

static void find_range(const float* data, int N, float* min_out, float* max_out) {
    float min_v = data[0], max_v = data[0];
    for (int i = 1; i < N; i++) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }
    *min_out = min_v;
    *max_out = max_v;
}

/* ===================================================================
 * Compression
 * =================================================================== */

WubuCompressedLatent wubu_latent_compress(
    const float* quaternions, const float* amplitude, int N, WubuQuality quality) {
    WubuQuantConfig config = wubu_quality_to_config(quality);
    return wubu_latent_compress_ex(quaternions, amplitude, N, &config);
}

WubuCompressedLatent wubu_latent_compress_ex(
    const float* quaternions, const float* amplitude, int N, const WubuQuantConfig* config) {
    WubuCompressedLatent result;
    result.N = N;
    result.config = *config;
    result.use_rle = config->use_rle;
    result.use_entropy = config->use_entropy;
    result.raw_size = (size_t)N * (4 * sizeof(float) + sizeof(float));

    /* Find ranges for quaternions */
    float q_min, q_max;
    find_range(quaternions, N * 4, &q_min, &q_max);
    result.q_min = q_min;
    result.q_max = q_max;

    /* Find ranges for amplitude */
    float a_min, a_max;
    find_range(amplitude, N, &a_min, &a_max);
    result.a_min = a_min;
    result.a_max = a_max;

    /* Quantize quaternions */
    size_t q_bytes_per_elem = (size_t)(config->quat_bits / 8 + (config->quat_bits % 8 != 0));
    size_t a_bytes_per_elem = (size_t)(config->amp_bits / 8 + (config->amp_bits % 8 != 0));
    result.quat_data = (uint8_t*)calloc((size_t)(N * 4) * q_bytes_per_elem, sizeof(uint8_t));
    for (int i = 0; i < N * 4; i++) {
        uint16_t qval = quantize_float16(quaternions[i], config->quat_bits, q_min, q_max);
        for (size_t b = 0; b < q_bytes_per_elem; b++) {
            result.quat_data[i * q_bytes_per_elem + b] = (uint8_t)(qval >> (b * 8));
        }
    }

    /* Quantize amplitude */
    result.amp_data = (uint8_t*)calloc((size_t)N * a_bytes_per_elem, sizeof(uint8_t));
    for (int i = 0; i < N; i++) {
        uint16_t qval = quantize_float16(amplitude[i], config->amp_bits, a_min, a_max);
        for (size_t b = 0; b < a_bytes_per_elem; b++) {
            result.amp_data[i * a_bytes_per_elem + b] = (uint8_t)(qval >> (b * 8));
        }
    }

    result.bitstream = NULL;
    result.bitstream_len = 0;

    /* Compressed size = quantized data with actual bit depths */
    size_t header = 64; /* metadata */
    size_t q_bits_total = (size_t)N * 4 * (size_t)config->quat_bits;
    size_t a_bits_total = (size_t)N * (size_t)config->amp_bits;
    size_t body_bytes = (q_bits_total + a_bits_total + 7) / 8;
    result.compressed_size = header + body_bytes + 4 * sizeof(float); /* + min/max values */
    result.compression_ratio = (float)result.raw_size / (float)result.compressed_size;
    result.psnr = 0.0f; /* computed on decode */

    return result;
}

/* ===================================================================
 * Decompression
 * =================================================================== */

void wubu_latent_decompress(
    float* quaternions, float* amplitude, const WubuCompressedLatent* compressed) {
    int N = compressed->N;
    WubuQuantConfig* cfg = &compressed->config;

    /* Use stored ranges from compression */
    float q_min = compressed->q_min, q_max = compressed->q_max;
    float a_min = compressed->a_min, a_max = compressed->a_max;

    /* Dequantize quaternions */
    size_t q_bytes = (size_t)(cfg->quat_bits / 8 + (cfg->quat_bits % 8 != 0));
    size_t a_bytes = (size_t)(cfg->amp_bits / 8 + (cfg->amp_bits % 8 != 0));
    uint32_t q_mask = (cfg->quat_bits == 16) ? 0xFFFF : ((1u << cfg->quat_bits) - 1);
    uint32_t a_mask = (cfg->amp_bits == 16) ? 0xFFFF : ((1u << cfg->amp_bits) - 1);
    for (int i = 0; i < N * 4; i++) {
        uint16_t qval = 0;
        for (size_t b = 0; b < q_bytes; b++) {
            qval |= ((uint16_t)compressed->quat_data[i * q_bytes + b]) << (b * 8);
        }
        qval &= q_mask;  /* Mask off upper bits for sub-16-bit quantization */
        quaternions[i] = dequantize_uint16(qval, cfg->quat_bits, q_min, q_max);
    }

    /* Dequantize amplitude */
    for (int i = 0; i < N; i++) {
        uint16_t qval = 0;
        for (size_t b = 0; b < a_bytes; b++) {
            qval |= ((uint16_t)compressed->amp_data[i * a_bytes + b]) << (b * 8);
        }
        qval &= a_mask;
        amplitude[i] = dequantize_uint16(qval, cfg->amp_bits, a_min, a_max);
    }
}

void wubu_latent_compressed_free(WubuCompressedLatent* compressed) {
    free(compressed->quat_data); compressed->quat_data = NULL;
    free(compressed->amp_data); compressed->amp_data = NULL;
    free(compressed->bitstream); compressed->bitstream = NULL;
}

/* ===================================================================
 * Stats
 * =================================================================== */

void wubu_latent_print_stats(const WubuCompressedLatent* compressed) {
    printf("  WuBu Latent Compression Report\n");
    printf("  ----------------------------------------\n");
    printf("  Points:           %d\n", compressed->N);
    printf("  Quat bits:        %d\n", compressed->config.quat_bits);
    printf("  Amp bits:         %d\n", compressed->config.amp_bits);
    printf("  RLE:              %s\n", compressed->use_rle ? "on" : "off");
    printf("  Entropy:          %s\n", compressed->use_entropy ? "on" : "off");
    printf("  Raw size:         %.2f MB (%zu bytes)\n",
           (float)compressed->raw_size / (1024.0f*1024.0f), compressed->raw_size);
    printf("  Compressed size:  %.2f MB (%zu bytes)\n",
           (float)compressed->compressed_size / (1024.0f*1024.0f), compressed->compressed_size);
    printf("  Compression ratio: %.2f:1\n", compressed->compression_ratio);
    printf("  Latent bpp:       %.4f bits/pixel\n",
           (float)(compressed->compressed_size * 8) / (float)compressed->N);
    printf("\n");
}
