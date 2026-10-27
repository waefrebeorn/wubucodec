/* wubu_latent_codec.c — WuBuLatent: Manifold-Native Latent Space Codec
 *
 * Encodes latent vectors that live on a Poincaré ball (hyperbolic manifold).
 * Philosophy: don't force hyperbolic latents into a Euclidean codec —
 * encode them where they live.
 *
 * Architecture:
 *   I-frame: Quantize base point on manifold (adaptive step from curvature)
 *   P-frame: Encode tangent velocity v = logmap(ref, curr) — Euclidean in tangent space
 *   B-frame: Geodesic interpolation between neighboring I/P frames
 *   Residuals: Quantized in tangent space, expmap back to manifold
 *
 * Reuses WUBV container format with new frame type for latents.
 *
 * Usage: wubu_latent_codec input.float32 output.wubv -d 64 -n 100 -c 1.0 -q 8
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

/* ============================ Hyperbolic Math ============================ */

#define EPS 1e-7f
#define TAN_VEC_CLAMP 10.0f

static float vec_norm_sq(const float* x, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += x[i] * x[i];
    return sum;
}

static float vec_norm(const float* x, int N) {
    return sqrtf(vec_norm_sq(x, N) + EPS);
}

static float vec_dot(const float* a, const float* b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

/* Clip to Poincaré ball interior */
static void poincare_clip(float* out, const float* x, int N, float c) {
    if (c <= 0.0f) { memcpy(out, x, N * sizeof(float)); return; }
    float sqrt_c = sqrtf(c + EPS);
    float max_norm = 1.0f / (sqrt_c + EPS) * (1.0f - EPS * 10.0f);
    float norm = vec_norm(x, N);
    if (norm > max_norm) {
        float scale = max_norm / (norm + EPS);
        for (int i = 0; i < N; i++) out[i] = x[i] * scale;
    } else {
        memcpy(out, x, N * sizeof(float));
    }
}

/* Exponential map: tangent space -> Poincaré ball at origin */
static void expmap(float* out, const float* v, int N, float c) {
    if (c <= 0.0f) { memcpy(out, v, N * sizeof(float)); return; }
    float sqrt_c = sqrtf(c + EPS);
    float v_norm = vec_norm(v, N);
    if (v_norm < EPS) { memcpy(out, v, N * sizeof(float)); return; }
    float arg = sqrt_c * v_norm;
    if (arg > 30.0f) arg = 30.0f;
    float tanh_val = tanhf(arg);
    float scale = tanh_val / (sqrt_c * v_norm + EPS);
    for (int i = 0; i < N; i++) out[i] = scale * v[i];
    poincare_clip(out, out, N, c);
}

/* Logarithmic map: Poincaré ball -> tangent space at origin */
static void logmap(float* out, const float* y, int N, float c) {
    if (c <= 0.0f) { memcpy(out, y, N * sizeof(float)); return; }
    float y_clipped[256];
    if (N > 256) {
        float* buf = (float*)malloc(N * sizeof(float));
        poincare_clip(buf, y, N, c);
        logmap(out, buf, N, c);
        free(buf);
        return;
    }
    poincare_clip(y_clipped, y, N, c);
    float sqrt_c = sqrtf(c + EPS);
    float y_norm = vec_norm(y_clipped, N);
    if (y_norm < EPS) { memcpy(out, y_clipped, N * sizeof(float)); return; }
    float atanh_arg = sqrt_c * y_norm;
    if (atanh_arg > 1.0f - EPS * 4.0f) atanh_arg = 1.0f - EPS * 4.0f;
    float atanh_val = atanhf(atanh_arg);
    float scale = atanh_val / (sqrt_c * y_norm + EPS);
    for (int i = 0; i < N; i++) out[i] = scale * y_clipped[i];
    float t_norm = vec_norm(out, N);
    if (t_norm > TAN_VEC_CLAMP) {
        float clamp_scale = TAN_VEC_CLAMP / t_norm;
        for (int i = 0; i < N; i++) out[i] *= clamp_scale;
    }
}

/* Möbius addition: x (+) y on Poincaré ball */
static void mobius_add(float* out, const float* x, const float* y, int N, float c) {
    if (c <= 0.0f) { for (int i = 0; i < N; i++) out[i] = x[i] + y[i]; return; }
    float x_norm_sq = vec_norm_sq(x, N);
    float y_norm_sq = vec_norm_sq(y, N);
    float xy_dot = vec_dot(x, y, N);
    float radius_sq = 1.0f / (c + EPS);
    if (x_norm_sq > radius_sq * (1.0f - EPS) * (1.0f - EPS))
        x_norm_sq = radius_sq * (1.0f - EPS) * (1.0f - EPS);
    if (y_norm_sq > radius_sq * (1.0f - EPS) * (1.0f - EPS))
        y_norm_sq = radius_sq * (1.0f - EPS) * (1.0f - EPS);
    float denom = 1.0f + 2.0f * c * xy_dot + c * c * x_norm_sq * y_norm_sq;
    if (denom < EPS) denom = EPS;
    float num_x = 1.0f + 2.0f * c * xy_dot + c * y_norm_sq;
    float num_y = 1.0f - c * x_norm_sq;
    for (int i = 0; i < N; i++) out[i] = (num_x * x[i] + num_y * y[i]) / denom;
    poincare_clip(out, out, N, c);
}

/* Exponential map at point x: exp_x(v) = x (+) expmap_0(v) */
static void expmap_at(float* out, const float* x, const float* v, int N, float c) {
    float exp_v[256];
    expmap(exp_v, v, N, c);
    mobius_add(out, x, exp_v, N, c);
}

/* Logarithmic map at point x: log_x(y) */
static void logmap_at(float* out, const float* x, const float* y, int N, float c) {
    /* log_x(y) = log_0((-x) (+) y) */
    float neg_x[256];
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];
    float rel[256];
    mobius_add(rel, neg_x, y, N, c);
    logmap(out, rel, N, c);
}

/* Geodesic interpolation: μ_t = exp_{x0}(t * log_{x0}(x1)) */
static void geodesic_interp(float* out, const float* x0, const float* x1, float t, int N, float c) {
    if (t <= 0.0f) { memcpy(out, x0, N * sizeof(float)); return; }
    if (t >= 1.0f) { memcpy(out, x1, N * sizeof(float)); return; }
    float tangent[256];
    logmap_at(tangent, x0, x1, N, c);
    for (int i = 0; i < N; i++) tangent[i] *= t;
    expmap_at(out, x0, tangent, N, c);
}

/* Hyperbolic distance */
static float hyperbolic_dist(const float* x, const float* y, int N, float c) {
    if (c <= 0.0f) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) { float d = x[i] - y[i]; sum += d * d; }
        return sqrtf(sum);
    }
    float neg_x[256];
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];
    float rel[256];
    mobius_add(rel, neg_x, y, N, c);
    float m_norm = vec_norm(rel, N);
    float sqrt_c = sqrtf(c + EPS);
    float arg = sqrt_c * m_norm;
    if (arg > 1.0f - EPS * 4.0f) arg = 1.0f - EPS * 4.0f;
    return (2.0f / sqrt_c) * atanhf(arg);
}

/* ============================ Quantization ============================ */

/* Adaptive QP based on curvature and position on manifold */
static float adaptive_qstep(float c, float norm, float base_qstep) {
    /* Near boundary: smaller QP (more bits) to avoid distortion */
    float sqrt_c = sqrtf(c + EPS);
    float max_norm = 1.0f / (sqrt_c + EPS);
    float proximity = norm / (max_norm + EPS);  /* 0 at center, 1 at boundary */
    /* Scale: 0.5x at center, 1.5x near boundary */
    float scale = 0.5f + proximity;
    return base_qstep * scale;
}

/* Quantize tangent vector with adaptive step */
static void quantize_tangent(int16_t* out, const float* v, int N, float qstep) {
    for (int i = 0; i < N; i++) {
        float scaled = v[i] / qstep;
        int level = (int)roundf(scaled);
        /* Clamp to int16 range */
        if (level > 32767) level = 32767;
        if (level < -32768) level = -32768;
        out[i] = (int16_t)level;
    }
}

/* Dequantize tangent vector */
static void dequantize_tangent(float* out, const int16_t* levels, int N, float qstep) {
    for (int i = 0; i < N; i++) {
        out[i] = (float)levels[i] * qstep;
    }
}

/* ============================ Bitstream ============================ */

typedef struct { uint8_t *buf; int byte_pos, bit_pos, size; } BitStream;

static void bs_init(BitStream* bs, uint8_t* buf, int size) {
    bs->buf = buf; bs->byte_pos = 0; bs->bit_pos = 0; bs->size = size;
}

static void bs_write_bit(BitStream* bs, int bit) {
    if (bs->byte_pos >= bs->size) return;
    if (bit) bs->buf[bs->byte_pos] |= (1 << (7 - bs->bit_pos));
    bs->bit_pos++;
    if (bs->bit_pos == 8) { bs->bit_pos = 0; bs->byte_pos++; }
}

static void bs_write_ue(BitStream* bs, int val) {
    int p = 0, t = val;
    while (t > 0) { bs_write_bit(bs, 1); t--; p++; }
    bs_write_bit(bs, 0);
    if (p > 0) for (int i = p-1; i >= 0; i--) bs_write_bit(bs, (val >> i) & 1);
}

static void bs_write_se(BitStream* bs, int val) {
    int a = abs(val), s = (val < 0) ? 1 : 0;
    bs_write_ue(bs, a);
    if (a > 0) bs_write_bit(bs, s);
}

static int bs_bytes(BitStream* bs) {
    return bs->byte_pos + (bs->bit_pos > 0 ? 1 : 0);
}

/* Encode a block of tangent vector coefficients */
static int encode_tangent_block(const int16_t* levels, uint8_t* out, int sz, int N) {
    BitStream bs; bs_init(&bs, out, sz);
    int nz = 0;
    for (int i = 0; i < N; i++) if (levels[i] != 0) nz++;
    bs_write_ue(&bs, nz);
    if (nz == 0) return bs_bytes(&bs);
    int run = 0;
    for (int i = 0; i < N; i++) {
        if (levels[i] != 0) { bs_write_ue(&bs, run); bs_write_se(&bs, levels[i]); run = 0; }
        else run++;
    }
    return bs_bytes(&bs);
}

/* ============================ Container ============================ */

/* WUBV container with latent payload:
 * Frame type 0 (KEY): I-frame latent (quantized base point)
 * Frame type 1 (INTER): P/B-frame tangent velocity
 *
 * Payload format:
 *   +0      2     dim (N)
 *   +2      1     num_vectors
 *   +3      1     frame_type (0=I, 1=P, 2=B)
 *   +4      ...   quantized data
 */

/* ============================ Main ============================ */

typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("WuBuLatent — Manifold-Native Latent Space Codec\n");
        printf("Usage: %s input.float32 output.wubv [options]\n", argv[0]);
        printf("  -d dim        Latent dimension (default 64)\n");
        printf("  -n num        Number of latent vectors (default 100)\n");
        printf("  -c curvature  Poincaré ball curvature (default 1.0)\n");
        printf("  -q qstep      Base quantization step (default 0.01)\n");
        printf("  -g gop        GOP size (default 10)\n");
        printf("  -b b_frames   B-frames between anchors (default 2)\n");
        printf("  -r refs       Reference frames (default 4)\n");
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int dim = 64, num_vecs = 100;
    float curvature = 1.0f, base_qstep = 0.01f;
    int gop_size = 10, b_frames = 2, nrefs = 4;

    for (int i = 3; i < argc; i += 2) {
        if (i+1 >= argc) break;
        if (strcmp(argv[i], "-d") == 0) dim = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-n") == 0) num_vecs = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-c") == 0) curvature = atof(argv[i+1]);
        else if (strcmp(argv[i], "-q") == 0) base_qstep = atof(argv[i+1]);
        else if (strcmp(argv[i], "-g") == 0) gop_size = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-b") == 0) b_frames = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-r") == 0) nrefs = atoi(argv[i+1]);
    }

    printf("WuBuLatent — Manifold-Native Latent Space Codec\n");
    printf("  Input:  %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Dim: %d, Vectors: %d, Curvature: %.2f\n", dim, num_vecs, curvature);
    printf("  QP: %.4f, GOP: %d, B-frames: %d\n", base_qstep, gop_size, b_frames);
    printf("\n");

    /* Load input */
    FILE* f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* Parse: either raw float32 [num_vecs, dim] or with header */
    int expected_size = num_vecs * dim * sizeof(float);
    if (fsz == expected_size) {
        /* Raw float32 data */
    } else if (fsz >= 8) {
        /* Try to read header */
        int header[2];
        if (fread(header, sizeof(int), 2, f) == 2) {
            num_vecs = header[0];
            dim = header[1];
            expected_size = num_vecs * dim * sizeof(float);
            if (fsz < expected_size + 8) {
                printf("Header says %d x %d but file too small\n", num_vecs, dim);
                return 1;
            }
            printf("  Read from header: %d vectors x %d dim\n", num_vecs, dim);
        } else {
            fseek(f, 0, SEEK_SET);
        }
    }

    expected_size = num_vecs * dim * sizeof(float);
    float* latents = (float*)malloc(expected_size);
    if (fread(latents, 1, expected_size, f) != (size_t)expected_size) {
        printf("Read failed (expected %d bytes)\n", expected_size);
        return 1;
    }
    fclose(f);
    printf("Loaded %d latent vectors (%d dim each)\n", num_vecs, dim);

    /* Project onto Poincaré ball (clip to interior) */
    for (int i = 0; i < num_vecs; i++) {
        float tmp[256];
        poincare_clip(tmp, latents + i * dim, dim, curvature);
        memcpy(latents + i * dim, tmp, dim * sizeof(float));
    }

    /* Determine frame types */
    FrameType* ftypes = (FrameType*)malloc(num_vecs * sizeof(FrameType));
    for (int i = 0; i < num_vecs; i++) {
        if (i % gop_size == 0) ftypes[i] = FRAME_I;
        else if (b_frames > 0 && (i % gop_size) % (b_frames + 1) != 0) ftypes[i] = FRAME_B;
        else ftypes[i] = FRAME_P;
    }

    /* Create container */
    WubuvWriter* wr = wubuv_writer_open(output, dim, num_vecs, 30, num_vecs, 0, 0, 1);
    if (!wr) { printf("Cannot create container\n"); return 1; }

    /* Working buffers */
    float* recon = (float*)calloc(num_vecs * dim, sizeof(float));
    float* all_recon = (float*)calloc(num_vecs * dim, sizeof(float));
    float* tangent = (float*)calloc(dim, sizeof(float));
    int16_t* quantized = (int16_t*)calloc(dim, sizeof(int16_t));

    double total_dist = 0;
    int ifr = 0, pfr = 0, bfr = 0;
    long total_comp = 0;

    /* Two-pass: Pass 1 for I/P, Pass 2 for B */
    printf("--- Pass 1: I/P frames ---\n");
    for (int i = 0; i < num_vecs; i++) {
        if (ftypes[i] == FRAME_B) continue;

        const float* curr = latents + i * dim;
        int is_i = (ftypes[i] == FRAME_I);

        /* Find reference */
        const float* ref = NULL;
        int ref_idx = i - 1;
        while (ref_idx >= 0 && ftypes[ref_idx] == FRAME_B) ref_idx--;
        if (ref_idx >= 0) ref = all_recon + ref_idx * dim;

        uint8_t* fbuf = (uint8_t*)malloc(dim * 4 + 64);
        int fpos = 0;

        /* Header */
        fbuf[0] = (uint8_t)(dim & 0xFF);
        fbuf[1] = (uint8_t)((dim >> 8) & 0xFF);
        fbuf[2] = 1;  /* num_vectors */
        fbuf[3] = (uint8_t)ftypes[i];
        fpos = 4;

        if (is_i || ref == NULL) {
            /* I-frame: quantize base point directly on manifold */
            float qstep = adaptive_qstep(curvature, vec_norm(curr, dim), base_qstep);
            /* Quantize in tangent space at origin */
            int16_t* iq = (int16_t*)calloc(dim, sizeof(int16_t));
            for (int d = 0; d < dim; d++) {
                float scaled = curr[d] / qstep;
                int level = (int)roundf(scaled);
                if (level > 32767) level = 32767;
                if (level < -32768) level = -32768;
                iq[d] = (int16_t)level;
            }
            /* Encode */
            uint8_t bb[2048];
            int bsz = encode_tangent_block(iq, bb, 2048, dim);
            memcpy(fbuf + fpos, bb, bsz);
            fpos += bsz;

            /* Reconstruct */
            for (int d = 0; d < dim; d++) {
                recon[i * dim + d] = (float)iq[d] * qstep;
            }
            free(iq);
            /* Project back to manifold */
            poincare_clip(recon + i * dim, recon + i * dim, dim, curvature);
            if (i % 10 == 0 || i < 5)
                printf("  Frame %d (I): qstep=%.4f, bytes=%d\n", i, qstep, bsz);
        } else {
            /* P-frame: encode tangent velocity */
            logmap_at(tangent, ref, curr, dim, curvature);
            float t_norm = vec_norm(tangent, dim);
            float qstep = adaptive_qstep(curvature, t_norm, base_qstep);

            quantize_tangent(quantized, tangent, dim, qstep);

            /* Encode */
            uint8_t bb[2048];
            int bsz = encode_tangent_block(quantized, bb, 2048, dim);
            memcpy(fbuf + fpos, bb, bsz);
            fpos += bsz;

            /* Reconstruct: expmap(ref, dequantized_tangent) */
            float dequantized[256];
            dequantize_tangent(dequantized, quantized, dim, qstep);
            expmap_at(recon + i * dim, ref, dequantized, dim, curvature);

            if (i % 10 == 0 || i < 5)
                printf("  Frame %d (P): qstep=%.4f, bytes=%d\n", i, qstep, bsz);
        }

        memcpy(all_recon + i * dim, recon + i * dim, dim * sizeof(float));
        wubuv_write_video(wr, fbuf, fpos, is_i);
        total_comp += fpos;
        if (is_i) ifr++; else pfr++;
        free(fbuf);
    }

    printf("--- Pass 2: B-frames ---\n");
    for (int i = 0; i < num_vecs; i++) {
        if (ftypes[i] != FRAME_B) continue;

        const float* curr = latents + i * dim;

        /* Find prev and next I/P frames */
        int prev_idx = i - 1; while (prev_idx >= 0 && ftypes[prev_idx] == FRAME_B) prev_idx--;
        int next_idx = i + 1; while (next_idx < num_vecs && ftypes[next_idx] == FRAME_B) next_idx++;

        const float* fwd_ref = (prev_idx >= 0) ? all_recon + prev_idx * dim : NULL;
        const float* bwd_ref = (next_idx < num_vecs) ? all_recon + next_idx * dim : fwd_ref;
        if (!fwd_ref) fwd_ref = bwd_ref;
        if (!bwd_ref) bwd_ref = fwd_ref;

        uint8_t* fbuf = (uint8_t*)malloc(dim * 4 + 64);
        int fpos = 0;

        /* Header */
        fbuf[0] = (uint8_t)(dim & 0xFF);
        fbuf[1] = (uint8_t)((dim >> 8) & 0xFF);
        fbuf[2] = 1;
        fbuf[3] = (uint8_t)FRAME_B;
        fpos = 4;

        /* Geodesic interpolation at t=0.5 */
        float interp[256];
        geodesic_interp(interp, fwd_ref, bwd_ref, 0.5f, dim, curvature);

        /* Encode residual: log_interp(curr) */
        float residual[256];
        logmap_at(residual, interp, curr, dim, curvature);
        float r_norm = vec_norm(residual, dim);
        float qstep = adaptive_qstep(curvature, r_norm, base_qstep * 1.5f);

        quantize_tangent(quantized, residual, dim, qstep);

        uint8_t bb[2048];
        int bsz = encode_tangent_block(quantized, bb, 2048, dim);
        memcpy(fbuf + fpos, bb, bsz);
        fpos += bsz;

        /* Reconstruct */
        float dequantized[256];
        dequantize_tangent(dequantized, quantized, dim, qstep);
        expmap_at(recon + i * dim, interp, dequantized, dim, curvature);

        memcpy(all_recon + i * dim, recon + i * dim, dim * sizeof(float));
        wubuv_write_video(wr, fbuf, fpos, 0);
        total_comp += fpos;
        bfr++;
        if (i % 10 == 0 || i < 5)
            printf("  Frame %d (B): qstep=%.4f, bytes=%d\n", i, qstep, bsz);
        free(fbuf);
    }

    wubuv_writer_close(wr);

    /* Compute distortion */
    for (int i = 0; i < num_vecs; i++) {
        float dist = hyperbolic_dist(latents + i * dim, all_recon + i * dim, dim, curvature);
        total_dist += dist;
    }
    double avg_dist = total_dist / num_vecs;

    printf("\n=== Results ===\n");
    printf("Frames: %d (I:%d P:%d B:%d)\n", num_vecs, ifr, pfr, bfr);
    printf("Average hyperbolic distance: %.6f\n", avg_dist);
    printf("Total: %.2f KB\n", total_comp / 1024.0);
    if (wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");

    free(latents); free(recon); free(all_recon);
    free(tangent); free(quantized); free(ftypes);
    return 0;
}
