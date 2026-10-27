/* wubuv_encode_v59.c — WuBuCodec v59: WuBuMath Hyperbolic Encoding
 *
 * Uses WuBuMath library for hyperbolic representation:
 * 1. Map 8×8 blocks to Poincaré ball (hyperbolic space)
 * 2. Multi-level nesting: 64-dim → 32-dim → 16-dim
 * 3. Quantize in tangent space (Euclidean, more efficient)
 * 4. Exploits hyperbolic geometry for natural compression
 * 5. CABAC entropy coding of quantized latents
 *
 * This replaces DCT with hyperbolic manifold representation.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_REF_FRAMES 4
#define BLOCK_SIZE 8
#define HYPER_DIM 64  /* Dimension of hyperbolic representation */

/* ============================ Hyperbolic Geometry ============================ */

typedef struct {
    float c;           /* curvature */
    float c_inv;       /* 1/sqrt(c) */
    float one_minus_c; /* (1 - c) */
} PoincareBall;

static float vdot(const float *a, const float *b, int N) {
    float s = 0.0f; for (int i = 0; i < N; i++) s += a[i] * b[i]; return s;
}
static float vnorm(const float *a, int N) { return sqrtf(vdot(a, a, N)); }

/* Poincaré ball exponential map: project tangent vector onto manifold */
static void poincare_exp(float *out, const float *p, const float *v, int N, float c) {
    if (c <= 0.0f) {
        for (int i = 0; i < N; i++) out[i] = p[i] + v[i];
        return;
    }
    float vn = vnorm(v, N);
    if (vn < 1e-8f) {
        memcpy(out, p, N * sizeof(float));
        return;
    }
    float p2 = vdot(p, p, N);
    float lam = 2.0f / (1.0f - c * p2 + 1e-8f);
    float t = tanhf(0.5f * lam * vn);
    float factor = t / (vn + 1e-8f);
    
    /* Möbius addition: p ⊕ v */
    float scaled[64];
    for (int i = 0; i < N; i++) scaled[i] = v[i] * factor;
    
    float xy = 0, x2 = 0, y2 = 0;
    for (int i = 0; i < N; i++) {
        xy += p[i] * scaled[i];
        x2 += p[i] * p[i];
        y2 += scaled[i] * scaled[i];
    }
    float den = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    float c1 = 1.0f + 2.0f * c * xy + c * y2;
    float c2 = 1.0f - c * x2;
    for (int i = 0; i < N; i++)
        out[i] = (c1 * p[i] + c2 * scaled[i]) / (den + 1e-8f);
}

/* Poincaré ball logarithmic map: project manifold point to tangent space */
static void poincare_log(float *out, const float *p, const float *x, int N, float c) {
    if (c <= 0.0f) {
        for (int i = 0; i < N; i++) out[i] = x[i] - p[i];
        return;
    }
    float negp[64], y[64];
    for (int i = 0; i < N; i++) negp[i] = -p[i];
    
    /* Möbius addition: (-p) ⊕ x */
    float xy = 0, x2 = 0, y2 = 0;
    for (int i = 0; i < N; i++) {
        xy += negp[i] * x[i];
        x2 += negp[i] * negp[i];
        y2 += x[i] * x[i];
    }
    float den = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    float c1 = 1.0f + 2.0f * c * xy + c * y2;
    float c2 = 1.0f - c * x2;
    for (int i = 0; i < N; i++)
        y[i] = (c1 * negp[i] + c2 * x[i]) / (den + 1e-8f);
    
    float yn = vnorm(y, N);
    if (yn < 1e-8f) {
        memset(out, 0, N * sizeof(float));
        return;
    }
    float p2 = vdot(p, p, N);
    float lam = 2.0f / (1.0f - c * p2 + 1e-8f);
    float at = atanhf(fminf(yn, 1.0f - 1e-8f));
    float factor = (2.0f / lam) * at / (yn + 1e-8f);
    for (int i = 0; i < N; i++) out[i] = y[i] * factor;
}

/* ============================ WuBuMath Encoder ============================ */

typedef struct {
    int num_levels;
    int hyperbolic_dims[4];
    float curvatures[4];
    float scales[4];
    
    /* Per-level origin points in hyperbolic space */
    float* level_origins[4];  /* [dim] per level */
    
    /* Quantization parameters */
    float quant_scale[4];
    int quant_bits[4];
} WubuEncoder;

static void wubu_init(WubuEncoder* enc) {
    enc->num_levels = 3;
    enc->hyperbolic_dims[0] = 64;
    enc->hyperbolic_dims[1] = 32;
    enc->hyperbolic_dims[2] = 16;
    enc->curvatures[0] = 0.5f;
    enc->curvatures[1] = 1.0f;
    enc->curvatures[2] = 2.0f;
    enc->scales[0] = 1.0f;
    enc->scales[1] = 0.7f;
    enc->scales[2] = 0.4f;
    enc->quant_bits[0] = 8;
    enc->quant_bits[1] = 6;
    enc->quant_bits[2] = 4;
    
    for (int i = 0; i < enc->num_levels; i++) {
        enc->level_origins[i] = (float*)calloc((size_t)enc->hyperbolic_dims[i], sizeof(float));
        /* Origin is at zero in Poincaré ball */
        enc->quant_scale[i] = 1.0f;
    }
}

/* Encode an 8×8 block to hyperbolic representation */
static void wubu_encode_block(WubuEncoder* enc, const uint8_t *block, 
                               float *latent, int *latent_size) {
    /* Step 1: Flatten 8×8 block to 64-dim vector */
    float tangent[64];
    for (int i = 0; i < 64; i++) {
        tangent[i] = (block[i] - 128.0f) / 128.0f; /* normalize to [-1, 1] */
    }
    
    /* Step 2: Map to hyperbolic space via exponential map */
    float origin[64] = {0}; /* origin in Poincaré ball */
    float hyperbolic[64];
    poincare_exp(hyperbolic, origin, tangent, 64, enc->curvatures[0]);
    
    /* Step 3: Multi-level nesting - project to lower dimensions */
    int offset = 0;
    for (int level = 0; level < enc->num_levels; level++) {
        int dim = enc->hyperbolic_dims[level];
        float scale = enc->scales[level];
        
        /* For level 0: use the 64-dim hyperbolic point */
        /* For higher levels: project to lower dim via log map + projection */
        float projected[64];
        if (level == 0) {
            memcpy(projected, hyperbolic, 64 * sizeof(float));
        } else {
            /* Log map to tangent space at origin */
            float tangent_prev[64];
            poincare_log(tangent_prev, enc->level_origins[level-1], hyperbolic, 
                        enc->hyperbolic_dims[level-1], enc->curvatures[level-1]);
            
            /* Project to lower dimension (simple truncation for now) */
            for (int i = 0; i < dim; i++) {
                projected[i] = tangent_prev[i * (enc->hyperbolic_dims[level-1] / dim)];
            }
            
            /* Exp map to new hyperbolic space */
            poincare_exp(hyperbolic, enc->level_origins[level], projected, dim, enc->curvatures[level]);
            memcpy(projected, hyperbolic, dim * sizeof(float));
        }
        
        /* Quantize in tangent space */
        float qs = (float)((1 << enc->quant_bits[level]) - 1);
        for (int i = 0; i < dim; i++) {
            /* Scale and quantize */
            float val = projected[i] * scale * qs;
            int qval = (int)(val + qs/2);
            if (qval < 0) qval = 0;
            if (qval >= (1 << enc->quant_bits[level])) qval = (1 << enc->quant_bits[level]) - 1;
            latent[offset + i] = (float)qval;
        }
        offset += dim;
    }
    *latent_size = offset;
}

static double rd_cost(long sse, int bits, double lambda) { return (double)sse + lambda * (double)bits; }

/* ============================ GOP Structure ============================ */

typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;
typedef struct { int gop_size; int b_frames; } GOPConfig;

static FrameType get_gop_type(int fn, GOPConfig *gop) {
    if (fn % gop->gop_size == 0) return FRAME_I;
    if (gop->b_frames > 0 && (fn % gop->gop_size) % (gop->b_frames + 1) != 0) return FRAME_B;
    return FRAME_P;
}

/* ============================ Motion Estimation ============================ */

static long sad_block(const uint8_t *cur, const uint8_t *ref, int W, int H, int bx, int by, int dx, int dy, int size) {
    int rx = bx + dx, ry = by + dy;
    if (rx < 0 || ry < 0 || rx+size > W || ry+size > H) return 1LL << 30;
    long s = 0;
    for (int r = 0; r < size; r++) for (int c = 0; c < size; c++)
        s += abs((int)cur[(by+r)*W+bx+c] - (int)ref[(ry+r)*W+rx+c]);
    return s;
}

static const int diamond[8][2] = {{-1,0},{0,-1},{0,1},{1,0},{-1,-1},{-1,1},{1,-1},{1,1}};

static void me_block(const uint8_t *cur, const uint8_t *ref, int W, int H, int bx, int by, int sr, int size, int *dx, int *dy, long *sad) {
    *dx = 0; *dy = 0; *sad = sad_block(cur, ref, W, H, bx, by, 0, 0, size);
    int step = sr;
    while (step > 0) {
        int imp = 1;
        while (imp) {
            imp = 0;
            for (int i = 0; i < 8; i++) {
                int nx = *dx + diamond[i][0]*step, ny = *dy + diamond[i][1]*step;
                if (abs(nx) > sr || abs(ny) > sr) continue;
                long s = sad_block(cur, ref, W, H, bx, by, nx, ny, size);
                if (s < *sad) { *sad = s; *dx = nx; *dy = ny; imp = 1; }
            }
        }
        step >>= 1;
    }
}

static void quarter_pel_interp(const uint8_t *ref, int W, int H, int bx, int by, int hdx, int hdy, int size, uint8_t *out) {
    for (int r = 0; r < size; r++) for (int c = 0; c < size; c++) {
        int x2 = (bx+c)*4 + hdx, y2 = (by+r)*4 + hdy;
        int x0 = x2/4, y0 = y2/4, x1 = x0+1, y1 = y0+1, fx = x2 & 3, fy = y2 & 3;
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 >= W) x1 = W-1; if (y1 >= H) y1 = H-1;
        out[r*size+c] = (uint8_t)((ref[y0*W+x0]*(4-fx)*(4-fy) + ref[y0*W+x1]*fx*(4-fy) + ref[y1*W+x0]*(4-fx)*fy + ref[y1*W+x1]*fx*fy) / 16);
    }
}

static void me_quarter_pel_block(const uint8_t *cur, const uint8_t *ref, int W, int H, int bx, int by, int sr, int size, int *out_hdx, int *out_hdy, long *out_sad) {
    int idx, idy; long isad;
    me_block(cur, ref, W, H, bx, by, sr, size, &idx, &idy, &isad);
    int best_hdx = idx*4, best_hdy = idy*4; long best_sad = isad;
    uint8_t hp_block[4096];
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int thdx = idx*4+dx, thdy = idy*4+dy;
        quarter_pel_interp(ref, W, H, bx, by, thdx, thdy, size, hp_block);
        long s = 0;
        for (int r = 0; r < size; r++) for (int c = 0; c < size; c++)
            s += abs((int)cur[(by+r)*W+bx+c] - (int)hp_block[r*size+c]);
        if (s < best_sad) { best_sad = s; best_hdx = thdx; best_hdy = thdy; }
    }
    *out_hdx = best_hdx; *out_hdy = best_hdy; *out_sad = best_sad;
}

static void mc_get_pred(const uint8_t *ref, int W, int H, int bx, int by, int hdx, int hdy, int size, uint8_t *pred) {
    quarter_pel_interp(ref, W, H, bx, by, hdx, hdy, size, pred);
}

/* ============================ CABAC ============================ */

#define CABAC_BITS 16
#define CABAC_ONE (1 << CABAC_BITS)
#define CABAC_HALF (1 << (CABAC_BITS - 1))

typedef struct {
    uint32_t low, range;
    int pending_bits;
    uint8_t byte_buffer;
    int byte_buffer_bits;
    uint8_t *output;
    int output_pos;
    int output_size;
    uint8_t ctx_mps[512];
    uint8_t ctx_state[512];
} CABACEncoder;

static const uint8_t transIdxMPS[64] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63
};

static const uint8_t transIdxLPS[64] = {
    0,0,1,2,2,4,4,5,6,7,8,9,9,11,11,12,
    13,13,15,15,16,16,18,18,19,20,20,21,22,22,23,24,
    24,25,26,26,27,27,28,28,29,29,30,30,30,31,32,32,
    33,33,33,34,34,35,35,35,36,36,36,37,37,37,38,38
};

static void cabac_init(CABACEncoder *enc, uint8_t *output, int size) {
    enc->low = 0;
    enc->range = CABAC_ONE - 1;
    enc->pending_bits = 0;
    enc->byte_buffer = 0;
    enc->byte_buffer_bits = 0;
    enc->output = output;
    enc->output_pos = 0;
    enc->output_size = size;
    for (int i = 0; i < 512; i++) { enc->ctx_mps[i] = 0; enc->ctx_state[i] = 32; }
}

static void cabac_write_bit(CABACEncoder *enc, int bit) {
    enc->byte_buffer = (enc->byte_buffer << 1) | bit;
    enc->byte_buffer_bits++;
    if (enc->byte_buffer_bits == 8) {
        if (enc->output_pos < enc->output_size) enc->output[enc->output_pos++] = enc->byte_buffer;
        enc->byte_buffer = 0;
        enc->byte_buffer_bits = 0;
    }
}

static void cabac_flush_bits(CABACEncoder *enc) {
    while (enc->byte_buffer_bits > 0) {
        enc->byte_buffer = (enc->byte_buffer << 1);
        enc->byte_buffer_bits++;
        if (enc->byte_buffer_bits == 8) {
            if (enc->output_pos < enc->output_size) enc->output[enc->output_pos++] = enc->byte_buffer;
            enc->byte_buffer = 0;
            enc->byte_buffer_bits = 0;
        }
    }
}

static void cabac_encode_bin(CABACEncoder *enc, int ctxIdx, int binVal) {
    uint8_t valMPS = enc->ctx_mps[ctxIdx];
    uint8_t pState = enc->ctx_state[ctxIdx];
    
    uint32_t rLPS = (pState < 32) ? (63 - pState) : pState;
    rLPS = (rLPS * enc->range) >> 8;
    if (rLPS < 2) rLPS = 2;
    uint32_t rMPS = enc->range - rLPS;
    
    if (binVal == valMPS) { enc->range = rMPS; }
    else { enc->low += rMPS; enc->range = rLPS; }
    
    if (binVal == valMPS) { enc->ctx_state[ctxIdx] = transIdxMPS[pState]; }
    else { enc->ctx_state[ctxIdx] = transIdxLPS[pState]; if (pState == 0) enc->ctx_mps[ctxIdx] = 1 - valMPS; }
    
    while (enc->range < CABAC_HALF) {
        if (enc->low >= CABAC_ONE) {
            enc->low -= CABAC_ONE;
            cabac_write_bit(enc, 1);
            while (enc->pending_bits > 0) { cabac_write_bit(enc, 0); enc->pending_bits--; }
        } else {
            cabac_write_bit(enc, 0);
            while (enc->pending_bits > 0) { cabac_write_bit(enc, 1); enc->pending_bits--; }
        }
        enc->range <<= 1;
        enc->low <<= 1;
    }
}

static void cabac_encode_ue(CABACEncoder *enc, int value, int ctxBase) {
    int prefix = 0, temp = value;
    while (temp > 0) { cabac_encode_bin(enc, ctxBase + prefix, 1); temp--; prefix++; }
    cabac_encode_bin(enc, ctxBase + prefix, 0);
    if (prefix > 0) for (int i = prefix - 1; i >= 0; i--)
        cabac_encode_bin(enc, ctxBase + 32 + i, (value >> i) & 1);
}

static void cabac_encode_se(CABACEncoder *enc, int value, int ctxBase) {
    int absVal = abs(value), sign = (value < 0) ? 1 : 0;
    cabac_encode_ue(enc, absVal, ctxBase);
    if (absVal > 0) cabac_encode_bin(enc, ctxBase + 64, sign);
}

static int cabac_finish(CABACEncoder *enc) {
    enc->range = CABAC_HALF / 2;
    enc->low += CABAC_HALF / 2;
    while (enc->range < CABAC_HALF) {
        if (enc->low >= CABAC_ONE) { enc->low -= CABAC_ONE; cabac_write_bit(enc, 1); }
        else cabac_write_bit(enc, 0);
        enc->range <<= 1;
        enc->low <<= 1;
    }
    cabac_flush_bits(enc);
    return enc->output_pos;
}

static double psnr_calc(const uint8_t *a, const uint8_t *b, int n) {
    double mse = 0; for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d*d; } mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

/* ============================ Main ============================ */

int main(int argc, char **argv) {
    if (argc < 3) { printf("Usage: %s input.raw output.wubv [options]\n", argv[0]); return 1; }

    const char *input = argv[1], *output = argv[2];
    int W = 320, H = 240, fps = 30, qp = 22, gop_size = 30, b_frames = 2;
    const char *recon_file = NULL;

    for (int i = 3; i < argc; i += 2) {
        if (i+1 >= argc) break;
        if (strcmp(argv[i], "-w") == 0) W = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-h") == 0) H = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-f") == 0) fps = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-q") == 0) qp = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-g") == 0) gop_size = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-b") == 0) b_frames = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--recon") == 0) { recon_file = argv[i+1]; }
    }

    int np = W * H;
    printf("WuBuCodec v59 — WuBuMath Hyperbolic Encoding (QP=%d)\n", qp);
    printf("  %dx%d @ %d fps, GOP=%d, B=%d\n", W, H, fps, qp, gop_size, b_frames);

    FILE *f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    int nf = (int)(fsz / np);
    uint8_t *frames = (uint8_t *)malloc((size_t)nf * np);
    if (fread(frames, 1, (size_t)nf*np, f) != (size_t)(nf*np)) { printf("Read failed\n"); return 1; }
    fclose(f);
    printf("Loaded %d frames\n", nf);

    WubuvWriter *wr = wubuv_writer_open(output, W, H, fps, nf, 0, 0, 1);
    if (!wr) { printf("Cannot create container\n"); return 1; }

    uint8_t *recon = (uint8_t *)malloc(np); memset(recon, 0, np);
    GOPConfig gop = {gop_size, b_frames};
    FrameType *ftypes = (FrameType *)malloc(nf * sizeof(FrameType));
    for (int fn = 0; fn < nf; fn++) ftypes[fn] = get_gop_type(fn, &gop);
    uint8_t *all_recon = (uint8_t *)calloc((size_t)nf * np, 1);

    double total_psnr = 0; int ifr = 0, pfr = 0, bfr = 0; long total_comp = 0;

    /* Initialize WuBuMath encoder */
    WubuEncoder wubu;
    wubu_init(&wubu);

    printf("--- Encoding ---\n");
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np; int is_i = (ftypes[fn] == FRAME_I);
        int is_b = (ftypes[fn] == FRAME_B);
        
        const uint8_t *ref_frames[NUM_REF_FRAMES] = {NULL, NULL, NULL, NULL};
        int ref_count = 0;
        for (int ref_fn = fn - 1; ref_fn >= 0 && ref_count < NUM_REF_FRAMES; ref_fn--) {
            if (ftypes[ref_fn] != FRAME_B) {
                ref_frames[ref_count++] = all_recon + ref_fn * np;
            }
        }

        uint8_t *fbuf = (uint8_t *)malloc(1024*1024); int fpos = 0;
        int frame_qp = is_b ? qp+1 : qp;
        
        CABACEncoder cabac;
        cabac_init(&cabac, fbuf, 1024*1024);
        
        int nbx = W/8, nby = H/8;
        for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
            int x0 = bx*8, y0 = by*8; int16_t blk[64]; 
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];
            double best_cost = 1e18; uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = 0; double lambda = 0.85 * pow(2.0, (frame_qp - 12.0) / 3.0);
            int best_transform = 0;

            /* Try inter prediction */
            for (int ref_i = 0; ref_i < ref_count; ref_i++) {
                int hdx, hdy; long sad;
                me_quarter_pel_block(orig, ref_frames[ref_i], W, H, x0, y0, 16, 8, &hdx, &hdy, &sad);
                uint8_t pred[64]; mc_get_pred(ref_frames[ref_i], W, H, x0, y0, hdx, hdy, 8, pred);
                
                /* WuBuMath: encode residual in hyperbolic space */
                int16_t res[64];
                for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                
                /* Map to hyperbolic space */
                float tangent[64], hyperbolic[64];
                for (int i = 0; i < 64; i++) tangent[i] = res[i] / 128.0f;
                float origin[64] = {0};
                poincare_exp(hyperbolic, origin, tangent, 64, 0.5f);
                
                /* Quantize in tangent space (more efficient than DCT coeffs) */
                float qs = (float)((1 << 8) - 1);
                int16_t lv[64];
                for (int i = 0; i < 64; i++) {
                    float val = hyperbolic[i] * qs;
                    int qval = (int)(val + qs/2);
                    if (qval < 0) qval = 0;
                    if (qval >= 256) qval = 255;
                    lv[i] = (int16_t)qval;
                }
                
                /* Dequantize */
                float dq_hyper[64];
                for (int i = 0; i < 64; i++) dq_hyper[i] = lv[i] / qs;
                
                /* Map back to pixel space via log map */
                float dq_tangent[64];
                poincare_log(dq_tangent, origin, dq_hyper, 64, 0.5f);
                
                int16_t rres[64];
                for (int i = 0; i < 64; i++) rres[i] = (int16_t)(dq_tangent[i] * 128.0f);
                
                long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                double cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = 8; best_transform = 0; }
            }

            /* Intra: direct hyperbolic encoding of block */
            uint8_t pred_i[64] = {0};
            float tangent[64], hyperbolic[64];
            for (int i = 0; i < 64; i++) tangent[i] = ((int)blk[i] - 128) / 128.0f;
            float origin[64] = {0};
            poincare_exp(hyperbolic, origin, tangent, 64, 0.5f);
            
            float qs = (float)((1 << 8) - 1);
            int16_t lv[64];
            for (int i = 0; i < 64; i++) {
                float val = hyperbolic[i] * qs;
                int qval = (int)(val + qs/2);
                if (qval < 0) qval = 0;
                if (qval >= 256) qval = 255;
                lv[i] = (int16_t)qval;
            }
            
            float dq_hyper[64];
            for (int i = 0; i < 64; i++) dq_hyper[i] = lv[i] / qs;
            
            float dq_tangent[64];
            poincare_log(dq_tangent, origin, dq_hyper, 64, 0.5f);
            
            int16_t rres[64];
            for (int i = 0; i < 64; i++) rres[i] = (int16_t)(dq_tangent[i] * 128.0f);
            
            long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)rres[r*8+c] + 128; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
            double cost = rd_cost(sse, 64, lambda);
            if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred_i, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = 8; best_transform = 0; }

            /* Reconstruct */
            float dq_hyper_loc[64];
            for (int i = 0; i < 64; i++) dq_hyper_loc[i] = best_lv[i] / qs;
            
            float dq_tangent_loc[64];
            poincare_log(dq_tangent_loc, origin, dq_hyper_loc, 64, 0.5f);
            
            int16_t rres_loc[64];
            for (int i = 0; i < 64; i++) rres_loc[i] = (int16_t)(dq_tangent_loc[i] * 128.0f);
            
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { 
                int v = (int)best_pred[r*8+c] + (int)rres_loc[r*8+c]; 
                recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); 
            }
            
            /* CABAC encode the latent */
            int16_t run_lv[64]; int run_count = 0;
            for (int i = 0; i < 64; i++) { if (best_lv[i] != 0) run_lv[run_count++] = best_lv[i]; }
            cabac_encode_ue(&cabac, run_count, 0);
            for (int i = 0; i < run_count; i++) cabac_encode_se(&cabac, run_lv[i], 10);
        }
        
        int cabac_bytes = cabac_finish(&cabac);
        double p = psnr_calc(orig, recon, np); total_psnr += p;
        memcpy(all_recon + fn * np, recon, np);
        wubuv_write_video(wr, fbuf, cabac_bytes, is_i); total_comp += cabac_bytes;
        if (fn % 30 == 0 || fn < 5) printf("  Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_i ? "I" : (ftypes[fn] == FRAME_B ? "B" : "P"), p, cabac_bytes);
        free(fbuf);
    }

    wubuv_writer_close(wr);
    double avg = total_psnr / nf;
    printf("\n=== Results ===\n"); printf("Frames: %d (I:%d P:%d B:%d)\n", nf, ifr, pfr, bfr);
    printf("Average PSNR: %.1f dB\n", avg); printf("Total: %.2f MB, bitrate: %.0f kbps\n", total_comp / 1048576.0, total_comp * 8.0 / nf * fps / 1024);
    if (wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");

    if (recon_file) { FILE* rf = fopen(recon_file, "wb"); if (rf) { fwrite(all_recon, 1, (size_t)nf * np, rf); fclose(rf); printf("Reconstruction saved to %s\n", recon_file); } }
    free(frames); free(recon); free(ftypes); free(all_recon);
    return 0;
}
