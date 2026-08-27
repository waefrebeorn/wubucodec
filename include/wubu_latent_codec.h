/*
 * wubu_latent_codec.h -- WuBu Latent Space Compression Codec
 *
 * Implements actual compression for Hamilton quaternion latent space:
 *   - Scalar quantization of quaternions (configurable bits)
 *   - Scalar quantization of amplitude (configurable bits)
 *   - Run-length encoding for zero-dominant channels
 *   - Entropy coding (frequency-based coding)
 *   - Bit-rate calculation and compression ratio reporting
 *
 * Target: 480P (854x480) = 409,920 points x (4 quaternions + 1 amplitude)
 *   Raw float: 409,920 x 5 x 4 = 8,198,400 bytes (~7.8 MB)
 *   Quantized 8-bit: 409,920 x 5 x 1 = 2,049,600 bytes (~1.95 MB) — 4:1
 *   Quantized 4-bit: 409,920 x 5 x 0.5 = 1,024,800 bytes (~1.0 MB) — 8:1
 *   With entropy coding: typically 2-3x additional compression
 */

#ifndef WUBU_LATENT_CODEC_H
#define WUBU_LATENT_CODEC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Quantization Configuration
 * =================================================================== */

typedef struct {
    int quat_bits;        /* bits per quaternion component (4-16) */
    int amp_bits;         /* bits per amplitude value (4-16) */
    int use_rle;          /* enable run-length encoding for zeros */
    int use_entropy;      /* enable entropy coding pass */
} WubuQuantConfig;

typedef enum {
    WUBU_QUALITY_LOSSLESS = 0,
    WUBU_QUALITY_HIGH    = 1,
    WUBU_QUALITY_MEDIUM  = 2,
    WUBU_QUALITY_LOW     = 3,
    WUBU_QUALITY_COUNT
} WubuQuality;

WubuQuantConfig wubu_quality_to_config(WubuQuality quality);

/* ===================================================================
 * Compressed Latent Data
 * =================================================================== */

typedef struct {
    uint8_t* quat_data;   /* [N * 4] quantized quaternion components */
    uint8_t* amp_data;    /* [N] quantized amplitude values */
    uint8_t* bitstream;   /* variable-length compressed data */
    size_t bitstream_len; /* bytes */
    int N;                /* number of points */
    int use_rle;
    int use_entropy;
    WubuQuantConfig config;
    float q_min, q_max;   /* quantization ranges for quaternions */
    float a_min, a_max;   /* quantization ranges for amplitude */
    size_t raw_size;
    size_t compressed_size;
    float compression_ratio;
    float psnr;
} WubuCompressedLatent;

/* ===================================================================
 * Encode / Decode
 * =================================================================== */

WubuCompressedLatent wubu_latent_compress(
    const float* quaternions, const float* amplitude, int N, WubuQuality quality);

WubuCompressedLatent wubu_latent_compress_ex(
    const float* quaternions, const float* amplitude, int N, const WubuQuantConfig* config);

void wubu_latent_decompress(
    float* quaternions, float* amplitude, const WubuCompressedLatent* compressed);

void wubu_latent_compressed_free(WubuCompressedLatent* compressed);

void wubu_latent_print_stats(const WubuCompressedLatent* compressed);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LATENT_CODEC_H */
