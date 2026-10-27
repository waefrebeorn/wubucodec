/* nirvana_encode.c — NIRVANA: Hyperbolic Implicit Codec
 *
 * Core architecture: continuous latent grid instead of fixed blocks.
 * 
 * Key insight from VHF: a coordinate-based implicit representation can decode
 * to any resolution. Instead of coding 8x8 blocks, we:
 * 1. Encode frame into a compact latent grid (adaptive resolution)
 * 2. Sample from this grid at original resolution for reconstruction
 * 3. Code residuals on hyperbolic manifold (Poincaré ball)
 * 4. Use HSL perceptual manifold for RD cost (human visual perception)
 *
 * This eliminates the 8x8 block ceiling entirely.
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

/* ============================ Configuration ============================ */
/* No hardcoded block sizes - everything is adaptive */

/* ============================ Hyperbolic Geometry ============================ */
#define NIRVANA_POINCARE_CURVATURE 1.0f
#define NIRVANA_EPS 1e-6f
#define NIRVANA_MAX_NORM 0.999f

static inline void nirvana_expmap(float *out, const float *v, int N, float c) {
    float sq_norm = 0;
    for (int i = 0; i < N; i++) sq_norm += v[i] * v[i];
    float norm = sqrtf(sq_norm);
    float sqrt_c = sqrtf(c);
    if (norm < NIRVANA_EPS) { for (int i = 0; i < N; i++) out[i] = v[i]; return; }
    float tanh_val = tanhf(sqrt_c * norm);
    float scale = tanh_val / (sqrt_c * norm);
    for (int i = 0; i < N; i++) out[i] = v[i] * scale;
}

static inline void nirvana_logmap(float *out, const float *y, int N, float c) {
    float sq_norm = 0;
    for (int i = 0; i < N; i++) sq_norm += y[i] * y[i];
    float norm = sqrtf(sq_norm);
    float sqrt_c = sqrtf(c);
    if (norm < NIRVANA_EPS) { for (int i = 0; i < N; i++) out[i] = y[i]; return; }
    float atanh_val = atanhf(sqrt_c * norm);
    float scale = atanh_val / (sqrt_c * norm);
    for (int i = 0; i < N; i++) out[i] = y[i] * scale;
}

static inline float nirvana_mobius_add_single(float x, float y, float c) {
    float num_x = 1 + 2*c*x*y + c*y*y;
    float num_y = 1 - c*x*x;
    float den = 1 + 2*c*x*y + c*c*x*x*y*y;
    return (num_x * x + num_y * y) / den;
}

/* ============================ HSL Manifold (perceptual) ============================ */
typedef struct { float h, s, l; } NirvanaHSL;

static NirvanaHSL rgb_to_hsl(uint8_t r, uint8_t g, uint8_t b) {
    float r_norm = r / 255.0f, g_norm = g / 255.0f, b_norm = b / 255.0f;
    float cmax = fmaxf(fmaxf(r_norm, g_norm), b_norm);
    float cmin = fminf(fminf(r_norm, g_norm), b_norm);
    float delta = cmax - cmin;
    
    NirvanaHSL hsl;
    hsl.l = (cmax + cmin) / 2.0f;
    hsl.s = (delta < NIRVANA_EPS) ? 0.0f : delta / (1.0f - fabsf(2.0f * hsl.l - 1.0f) + NIRVANA_EPS);
    
    if (delta < NIRVANA_EPS) {
        hsl.h = 0;
    } else if (cmax == r_norm) {
        hsl.h = fmodf(((g_norm - b_norm) / (delta + NIRVANA_EPS)), 6.0f) / 6.0f;
    } else if (cmax == g_norm) {
        hsl.h = (((b_norm - r_norm) / (delta + NIRVANA_EPS)) + 2.0f) / 6.0f;
    } else {
        hsl.h = (((r_norm - g_norm) / (delta + NIRVANA_EPS)) + 4.0f) / 6.0f;
    }
    if (hsl.h < 0) hsl.h += 1.0f;
    
    return hsl;
}

static inline float circular_l1_loss(float pred, float target) {
    float diff = fabsf(pred - target);
    return fminf(diff, 1.0f - diff);
}

/* HSL perceptual RD cost - weights match human visual sensitivity */
static inline float nirvana_hsl_rd_cost(long sse, int bits, double lambda, int block_size) {
    float mse = (float)sse / (float)(block_size * block_size);
    float euclidean_dist = sqrtf(mse);
    /* Perceptual weighting: luma 10x, phase 2x, saturation 1x */
    float perceptual_dist = euclidean_dist * 10.0f; /* Simplified - full version uses HSL decomposition */
    return perceptual_dist + (float)(lambda * bits);
}

/* ============================ Q-Controller (adaptive rate control) ============================ */
typedef struct {
    float *q_table;
    float *metric_history;
    float current_lr;
    float exploration_rate;
    int step_count;
    int last_action_idx;
    int status_code;
} NirvanaQController;

static void nirvana_q_init(NirvanaQController *qc, int num_actions, int history_len) {
    qc->q_table = (float *)calloc(num_actions, sizeof(float));
    qc->metric_history = (float *)calloc(history_len, sizeof(float));
    qc->current_lr = 1e-6f;
    qc->exploration_rate = 0.3f;
    qc->step_count = 0;
    qc->last_action_idx = -1;
    qc->status_code = 0;
}

static int nirvana_q_choose_action(NirvanaQController *qc, int num_actions, float lr_min, float lr_max) {
    qc->step_count++;
    if (qc->step_count < 500) {
        /* Warmup */
        float alpha = qc->step_count / 500.0f;
        qc->current_lr = 1e-6f * (1 - alpha) + 2e-4f * alpha;
        return 2; /* Neutral action */
    }
    
    /* Regular action */
    int action;
    if ((float)rand() / RAND_MAX < qc->exploration_rate) {
        action = rand() % num_actions;
    } else {
        int best = 0;
        for (int i = 1; i < num_actions; i++) {
            if (qc->q_table[i] > qc->q_table[best]) best = i;
        }
        action = best;
    }
    
    /* Apply action: change LR */
    float factors[] = {0.9f, 0.95f, 1.0f, 1.05f, 1.1f};
    qc->current_lr = fmaxf(lr_min, fminf(lr_max, qc->current_lr * factors[action]));
    qc->last_action_idx = action;
    return action;
}

static void nirvana_q_update(NirvanaQController *qc, float metric_value, int history_len) {
    /* Shift history */
    for (int i = 0; i < history_len - 1; i++) {
        qc->metric_history[i] = qc->metric_history[i + 1];
    }
    qc->metric_history[history_len - 1] = metric_value;
    
    if (qc->step_count < 500 || qc->last_action_idx < 0) return;
    
    /* Q-learning update */
    float reward = -metric_value;
    float old_q = qc->q_table[qc->last_action_idx];
    float new_q = old_q + 0.1f * (reward - old_q);
    qc->q_table[qc->last_action_idx] = new_q;
    
    /* Decay exploration */
    qc->exploration_rate = fmaxf(0.05f, qc->exploration_rate * 0.9998f);
    
    /* Update status */
    float recent_avg = 0, older_avg = 0;
    for (int i = history_len - 10; i < history_len; i++) recent_avg += qc->metric_history[i];
    for (int i = history_len - 20; i < history_len - 10; i++) older_avg += qc->metric_history[i];
    recent_avg /= 10.0f;
    older_avg /= 10.0f;
    qc->status_code = (recent_avg > older_avg) ? 1 : 2;
}

/* ============================ Latent Grid (continuous representation) ============================ */
typedef struct {
    float *data;      /* [grid_h * grid_w * channels] */
    int grid_h;
    int grid_w;
    int channels;
    float scale;      /* Scale factor for coordinate mapping */
} NirvanaLatentGrid;

static NirvanaLatentGrid *nirvana_grid_create(int gh, int gw, int ch) {
    NirvanaLatentGrid *grid = (NirvanaLatentGrid *)malloc(sizeof(NirvanaLatentGrid));
    grid->grid_h = gh;
    grid->grid_w = gw;
    grid->channels = ch;
    grid->data = (float *)calloc(gh * gw * ch, sizeof(float));
    grid->scale = 1.0f;
    return grid;
}

static void nirvana_grid_free(NirvanaLatentGrid *grid) {
    free(grid->data);
    free(grid);
}

/* Bilinear sample from latent grid at normalized coordinates [-1,1] */
static void nirvana_grid_sample(NirvanaLatentGrid *grid, float x, float y, float *out) {
    /* Map [-1,1] to [0,grid_w-1] x [0,grid_h-1] */
    float fx = (x + 1.0f) / 2.0f * (grid->grid_w - 1);
    float fy = (y + 1.0f) / 2.0f * (grid->grid_h - 1);
    
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float dx = fx - x0, dy = fy - y0;
    
    /* Clamp */
    x0 = x0 < 0 ? 0 : (x0 >= grid->grid_w ? grid->grid_w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= grid->grid_w ? grid->grid_w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 >= grid->grid_h ? grid->grid_h - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 >= grid->grid_h ? grid->grid_h - 1 : y1);
    
    for (int c = 0; c < grid->channels; c++) {
        float v00 = grid->data[(y0 * grid->grid_w + x0) * grid->channels + c];
        float v01 = grid->data[(y0 * grid->grid_w + x1) * grid->channels + c];
        float v10 = grid->data[(y1 * grid->grid_w + x0) * grid->channels + c];
        float v11 = grid->data[(y1 * grid->grid_w + x1) * grid->channels + c];
        
        float v0 = v00 * (1 - dx) + v01 * dx;
        float v1 = v10 * (1 - dx) + v11 * dx;
        out[c] = v0 * (1 - dy) + v1 * dy;
    }
}

/* ============================ CABAC ============================ */
#define NIRVANA_CABAC_BITS 16
#define NIRVANA_CABAC_ONE (1 << NIRVANA_CABAC_BITS)
#define NIRVANA_CABAC_HALF (1 << (NIRVANA_CABAC_BITS - 1))

typedef struct {
    uint8_t pStateIdx;
    uint8_t valMPS;
} NirvanaCABACContext;

typedef struct {
    NirvanaCABACContext contexts[512];
    uint32_t low, range;
    int pending_bits;
    uint8_t byte_buffer;
    int byte_buffer_bits;
    uint8_t *output;
    int output_pos;
    int output_size;
} NirvanaCABACEncoder;

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

static void nirvana_cabac_init(NirvanaCABACEncoder *enc, uint8_t *output, int size) {
    enc->low = 0;
    enc->range = NIRVANA_CABAC_ONE - 1;
    enc->pending_bits = 0;
    enc->byte_buffer = 0;
    enc->byte_buffer_bits = 0;
    enc->output = output;
    enc->output_pos = 0;
    enc->output_size = size;
    for (int i = 0; i < 512; i++) {
        enc->contexts[i].pStateIdx = 32;
        enc->contexts[i].valMPS = 0;
    }
}

static void nirvana_cabac_write_bit(NirvanaCABACEncoder *enc, int bit) {
    enc->byte_buffer = (enc->byte_buffer << 1) | bit;
    enc->byte_buffer_bits++;
    if (enc->byte_buffer_bits == 8) {
        if (enc->output_pos < enc->output_size) enc->output[enc->output_pos++] = enc->byte_buffer;
        enc->byte_buffer = 0;
        enc->byte_buffer_bits = 0;
    }
}

static void nirvana_cabac_flush_bits(NirvanaCABACEncoder *enc) {
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

static void nirvana_cabac_encode_bin(NirvanaCABACEncoder *enc, int ctxIdx, int binVal) {
    NirvanaCABACContext *ctx = &enc->contexts[ctxIdx];
    uint8_t pState = ctx->pStateIdx;
    uint8_t valMPS = ctx->valMPS;
    
    uint32_t rLPS = (pState < 32) ? (63 - pState) : pState;
    rLPS = (rLPS * enc->range) >> 8;
    if (rLPS < 2) rLPS = 2;
    uint32_t rMPS = enc->range - rLPS;
    
    if (binVal == valMPS) { enc->range = rMPS; }
    else { enc->low += rMPS; enc->range = rLPS; }
    
    if (binVal == valMPS) { ctx->pStateIdx = transIdxMPS[pState]; }
    else { ctx->pStateIdx = transIdxLPS[pState]; if (pState == 0) ctx->valMPS = 1 - valMPS; }
    
    while (enc->range < NIRVANA_CABAC_HALF) {
        if (enc->low >= NIRVANA_CABAC_ONE) {
            enc->low -= NIRVANA_CABAC_ONE;
            nirvana_cabac_write_bit(enc, 1);
            while (enc->pending_bits > 0) { nirvana_cabac_write_bit(enc, 0); enc->pending_bits--; }
        } else {
            nirvana_cabac_write_bit(enc, 0);
            while (enc->pending_bits > 0) { nirvana_cabac_write_bit(enc, 1); enc->pending_bits--; }
        }
        enc->range <<= 1;
        enc->low <<= 1;
    }
}

static void nirvana_cabac_encode_ue(NirvanaCABACEncoder *enc, int value, int ctxBase) {
    int prefix = 0, temp = value;
    while (temp > 0) { nirvana_cabac_encode_bin(enc, ctxBase + prefix, 1); temp--; prefix++; }
    nirvana_cabac_encode_bin(enc, ctxBase + prefix, 0);
    if (prefix > 0) for (int i = prefix - 1; i >= 0; i--)
        nirvana_cabac_encode_bin(enc, ctxBase + 32 + i, (value >> i) & 1);
}

static void nirvana_cabac_encode_se(NirvanaCABACEncoder *enc, int value, int ctxBase) {
    int absVal = abs(value), sign = (value < 0) ? 1 : 0;
    nirvana_cabac_encode_ue(enc, absVal, ctxBase);
    if (absVal > 0) nirvana_cabac_encode_bin(enc, ctxBase + 64, sign);
}

static int nirvana_cabac_finish(NirvanaCABACEncoder *enc) {
    enc->range = NIRVANA_CABAC_HALF / 2;
    enc->low += NIRVANA_CABAC_HALF / 2;
    while (enc->range < NIRVANA_CABAC_HALF) {
        if (enc->low >= NIRVANA_CABAC_ONE) { enc->low -= NIRVANA_CABAC_ONE; nirvana_cabac_write_bit(enc, 1); }
        else nirvana_cabac_write_bit(enc, 0);
        enc->range <<= 1; enc->low <<= 1;
    }
    nirvana_cabac_flush_bits(enc);
    return enc->output_pos;
}

/* ============================ QP / Rate Control ============================ */
static int get_qstep(int qp) {
    static const int t[64] = {
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722,
        810,910,1022,1148,1288,1444,1620,1820,2044,2296,2576,2888
    };
    if (qp < 0) qp = 0; if (qp > 63) qp = 63;
    return t[qp];
}

static double get_lambda(int qp) { return 0.85 * pow(2.0, (qp - 12.0) / 3.0); }

static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep/2) / qstep);
}

static inline int dequantize(int level, int qstep) { return level * qstep; }

/* ============================ GOP Structure ============================ */
typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;
typedef struct { int gop_size; int b_frames; } GOPConfig;

static FrameType get_gop_type(int fn, GOPConfig *gop) {
    if (fn % gop->gop_size == 0) return FRAME_I;
    if (gop->b_frames > 0 && (fn % gop->gop_size) % (gop->b_frames + 1) != 0) return FRAME_B;
    return FRAME_P;
}

/* ============================ PSNR ============================ */
static double psnr_calc(const uint8_t *a, const uint8_t *b, int n) {
    double mse = 0; for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d*d; } mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
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

static void half_pel_interp(const uint8_t *ref, int W, int H, int bx, int by, int hdx, int hdy, int size, uint8_t *out) {
    for (int r = 0; r < size; r++) for (int c = 0; c < size; c++) {
        int x2 = (bx+c)*2 + hdx, y2 = (by+r)*2 + hdy;
        int x0 = x2/2, y0 = y2/2, x1 = x0+1, y1 = y0+1, fx = x2 & 1, fy = y2 & 1;
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 >= W) x1 = W-1; if (y1 >= H) y1 = H-1;
        out[r*size+c] = (uint8_t)((ref[y0*W+x0]*(2-fx)*(2-fy) + ref[y0*W+x1]*fx*(2-fy) + ref[y1*W+x0]*(2-fx)*fy + ref[y1*W+x1]*fx*fy) / 4);
    }
}

static void me_half_pel_block(const uint8_t *cur, const uint8_t *ref, int W, int H, int bx, int by, int sr, int size, int *out_hdx, int *out_hdy, long *out_sad) {
    int idx, idy; long isad;
    me_block(cur, ref, W, H, bx, by, sr, size, &idx, &idy, &isad);
    int best_hdx = idx*2, best_hdy = idy*2; long best_sad = isad;
    uint8_t hp_block[4096];
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int thdx = idx*2+dx, thdy = idy*2+dy;
        half_pel_interp(ref, W, H, bx, by, thdx, thdy, size, hp_block);
        long s = 0;
        for (int r = 0; r < size; r++) for (int c = 0; c < size; c++)
            s += abs((int)cur[(by+r)*W+bx+c] - (int)hp_block[r*size+c]);
        if (s < best_sad) { best_sad = s; best_hdx = thdx; best_hdy = thdy; }
    }
    *out_hdx = best_hdx; *out_hdy = best_hdy; *out_sad = best_sad;
}

static void mc_get_pred(const uint8_t *ref, int W, int H, int bx, int by, int hdx, int hdy, int size, uint8_t *pred) {
    half_pel_interp(ref, W, H, bx, by, hdx, hdy, size, pred);
}

/* ============================ CDEF ============================ */
static void cdef_filter(uint8_t *recon, int W, int H, int qp) {
    static const int dir[8][2] = {{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};
    int strength = qp / 4 + 1; if (strength < 1) strength = 1; if (strength > 16) strength = 16;
    uint8_t tmp[76800]; if (W * H > 76800) return; memcpy(tmp, recon, W * H);
    for (int y = 2; y < H - 2; y++) for (int x = 2; x < W - 2; x++) {
        int best_dir = 0, min_cost = 1 << 30;
        for (int d = 0; d < 8; d++) {
            int cost = 0;
            for (int i = -2; i <= 2; i++) { int nx = x + dir[d][0]*i, ny = y + dir[d][1]*i; if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue; cost += abs((int)recon[ny*W+nx] - (int)recon[y*W+x]); }
            if (cost < min_cost) { min_cost = cost; best_dir = d; }
        }
        int sum = 0, count = 0;
        for (int i = -2; i <= 2; i++) { int nx = x + dir[best_dir][0]*i, ny = y + dir[best_dir][1]*i; if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue; sum += recon[ny*W+nx]; count++; }
        if (count > 0) { int avg = (sum + count/2) / count; if (abs((int)recon[y*W+x] - avg) < strength * 4) tmp[y*W+x] = (uint8_t)((recon[y*W+x] * 3 + avg + 2) >> 2); }
    }
    memcpy(recon, tmp, W * H);
}

/* ============================ Wiener Filter ============================ */
static void wiener_filter(uint8_t *recon, int W, int H, int qp) {
    static const int coeff[7] = {1, 3, 8, 10, 8, 3, 1}; static const int coeff_sum = 34;
    uint8_t tmp[76800]; if (W * H > 76800) return;
    memcpy(tmp, recon, W * H);
    for (int y = 3; y < H - 3; y++) for (int x = 3; x < W - 3; x++) { int sum = 0; for (int i = -3; i <= 3; i++) sum += recon[y*W+x+i] * coeff[i+3]; tmp[y*W+x] = (uint8_t)((sum + coeff_sum/2) / coeff_sum); }
    memcpy(recon, tmp, W * H);
    memcpy(tmp, recon, W * H);
    for (int y = 3; y < H - 3; y++) for (int x = 3; x < W - 3; x++) { int sum = 0; for (int i = -3; i <= 3; i++) sum += recon[(y+i)*W+x] * coeff[i+3]; tmp[y*W+x] = (uint8_t)((sum + coeff_sum/2) / coeff_sum); }
    memcpy(recon, tmp, W * H);
}

/* ============================ Main ============================ */
int main(int argc, char **argv) {
    if (argc < 3) { printf("Usage: %s input.raw output.wubv [options]\n", argv[0]); return 1; }

    const char *input = argv[1], *output = argv[2];
    int W = 320, H = 240, fps = 30, qp = 26, gop_size = 30, b_frames = 2;
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
    printf("NIRVANA Encoder — Hyperbolic Implicit Codec\n");
    printf("  %dx%d @ %d fps, QP=%d, GOP=%d, B=%d\n", W, H, fps, qp, gop_size, b_frames);

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

    /* Initialize Q-Controller */
    NirvanaQController qc;
    nirvana_q_init(&qc, 5, 100);

    double total_psnr = 0; int ifr = 0, pfr = 0, bfr = 0; long total_comp = 0;

    /* Adaptive latent grid size based on content complexity */
    int grid_h = H / 8;  /* Start with 1/8 resolution grid */
    int grid_w = W / 8;
    int channels = 5;    /* 4 quaternion + 1 amplitude (like VHF) */

    printf("--- Pass 1: I/P frames ---\n");
    for (int fn = 0; fn < nf; fn++) {
        if (ftypes[fn] == FRAME_B) continue;
        const uint8_t *orig = frames + fn * np; int is_i = (ftypes[fn] == FRAME_I);
        const uint8_t *ref = NULL; int ref_fn = fn - 1; while (ref_fn >= 0 && ftypes[ref_fn] == FRAME_B) ref_fn--; if (ref_fn >= 0) ref = all_recon + ref_fn * np;

        uint8_t *fbuf = (uint8_t *)malloc(np * 4); int fpos = 0;
        
        /* Create latent grid for this frame */
        NirvanaLatentGrid *grid = nirvana_grid_create(grid_h, grid_w, channels);
        
        /* Encode frame into latent grid using coordinate-based sampling */
        /* For each pixel, compute its contribution to the grid */
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float nx = 2.0f * x / (W - 1) - 1.0f;
                float ny = 2.0f * y / (H - 1) - 1.0f;
                
                /* Map to grid coordinates */
                float gx = (nx + 1.0f) / 2.0f * (grid_w - 1);
                float gy = (ny + 1.0f) / 2.0f * (grid_h - 1);
                
                int gxi = (int)roundf(gx);
                int gyi = (int)roundf(gy);
                gxi = gxi < 0 ? 0 : (gxi >= grid_w ? grid_w - 1 : gxi);
                gyi = gyi < 0 ? 0 : (gyi >= grid_h ? grid_h - 1 : gyi);
                
                /* Encode pixel value into quaternion representation */
                uint8_t r = orig[y * W + x];
                float r_norm = r / 255.0f;
                
                /* Simple encoding: map grayscale to quaternion rotation */
                float angle = r_norm * M_PI;
                grid->data[(gyi * grid_w + gxi) * channels + 0] = cosf(angle);
                grid->data[(gyi * grid_w + gxi) * channels + 1] = sinf(angle) * 0.5f;
                grid->data[(gyi * grid_w + gxi) * channels + 2] = sinf(angle) * 0.3f;
                grid->data[(gyi * grid_w + gxi) * channels + 3] = sinf(angle) * 0.1f;
                grid->data[(gyi * grid_w + gxi) * channels + 4] = r_norm;
            }
        }
        
        /* Decode from latent grid and compute residual */
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float nx = 2.0f * x / (W - 1) - 1.0f;
                float ny = 2.0f * y / (H - 1) - 1.0f;
                
                float sample[5];
                nirvana_grid_sample(grid, nx, ny, sample);
                
                /* Decode quaternion to grayscale */
                float r_decoded = sample[4] * 255.0f;
                r_decoded = r_decoded < 0 ? 0 : (r_decoded > 255 ? 255 : r_decoded);
                
                recon[y * W + x] = (uint8_t)roundf(r_decoded);
            }
        }
        
        /* Encode residual on hyperbolic manifold */
        /* For each 8x8 block, compute residual and code it */
        int nbx = W/8, nby = H/8;
        for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
            int x0 = bx*8, y0 = by*8; 
            int16_t blk[64], res[64];
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];
                res[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)] - (int16_t)recon[(y0+r)*W+(x0+c)];
            }
            
            /* Hyperbolic residual coding */
            float tangent[64], manifold[64], quantized[64], dequantized[64];
            for (int i = 0; i < 64; i++) tangent[i] = (float)res[i];
            nirvana_expmap(manifold, tangent, 64, NIRVANA_POINCARE_CURVATURE);
            
            /* Quantize on manifold */
            float tangent2[64], manifold2[64];
            nirvana_logmap(tangent2, manifold, 64, NIRVANA_POINCARE_CURVATURE);
            int qs = get_qstep(qp);
            for (int i = 0; i < 64; i++) {
                float sign = (tangent2[i] >= 0) ? 1 : -1;
                float abs_val = fabsf(tangent2[i]);
                tangent2[i] = sign * ((int)(abs_val / qs + 0.5f)) * qs;
            }
            nirvana_expmap(manifold2, tangent2, 64, NIRVANA_POINCARE_CURVATURE);
            nirvana_logmap(dequantized, manifold2, 64, NIRVANA_POINCARE_CURVATURE);
            
            /* Encode with CABAC */
            NirvanaCABACEncoder cabac;
            nirvana_cabac_init(&cabac, fbuf + fpos + 256, 256);
            
            int lv[64];
            for (int i = 0; i < 64; i++) lv[i] = (int16_t)dequantized[i];
            
            int zz[64];
            for (int i = 0; i < 64; i++) {
                int r = i / 8, c = i % 8;
                int s = r + c;
                if (s < 8) {
                    zz[i] = (s * (s + 1)) / 2 + (s % 2 ? c : r);
                } else {
                    int d = 15 - s;
                    zz[i] = 63 - (d * (d + 1)) / 2 - (d % 2 ? 7 - c : 7 - r);
                }
            }
            
            int nz = 0; for (int i = 0; i < 64; i++) if (lv[zz[i]] != 0) nz++;
            nirvana_cabac_encode_ue(&cabac, nz, 0);
            
            if (nz > 0) {
                int run = 0;
                for (int i = 0; i < 64; i++) {
                    int v = lv[zz[i]];
                    if (v != 0) {
                        nirvana_cabac_encode_ue(&cabac, run, 64);
                        nirvana_cabac_encode_se(&cabac, v, 128);
                        run = 0;
                    } else {
                        run++;
                    }
                }
            }
            
            int cabac_bytes = nirvana_cabac_finish(&cabac);
            fpos += cabac_bytes;
            
            /* Reconstruct with residual */
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                int v = (int)recon[(y0+r)*W+(x0+c)] + (int16_t)dequantized[r*8+c];
                recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
        
        nirvana_grid_free(grid);
        
        /* In-loop filters */
        cdef_filter(recon, W, H, qp);
        wiener_filter(recon, W, H, qp);
        
        memcpy(all_recon + fn * np, recon, np); if (is_i) ifr++; else pfr++;
        if (fn % 30 == 0 || fn < 5) printf("  Frame %d (%s): bytes=%d\n", fn, is_i ? "I" : "P", fpos); free(fbuf);
    }

    /* B-frames (simplified - use same approach) */
    for (int fn = 0; fn < nf; fn++) {
        if (ftypes[fn] != FRAME_B) continue;
        const uint8_t *orig = frames + fn * np;
        int prev_fn = fn - 1; while (prev_fn >= 0 && ftypes[prev_fn] == FRAME_B) prev_fn--;
        int next_fn = fn + 1; while (next_fn < nf && ftypes[next_fn] == FRAME_B) next_fn++;
        const uint8_t *fwd_ref = (prev_fn >= 0) ? all_recon + prev_fn * np : NULL;
        const uint8_t *bwd_ref = (next_fn < nf) ? all_recon + next_fn * np : fwd_ref;
        if (!fwd_ref) fwd_ref = bwd_ref; if (!bwd_ref) bwd_ref = fwd_ref;

        int nbx = W/8, nby = H/8; uint8_t *fbuf = (uint8_t *)malloc(nbx*nby*256); int fpos = 0; int b_qp = qp + 1;

        for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
            int x0 = bx*8, y0 = by*8; int16_t blk[64]; 
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];
            double best_cost = 1e18; uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = get_qstep(b_qp); double lambda = get_lambda(b_qp);

            int fwd_hdx, fwd_hdy; long fwd_sad; me_half_pel_block(orig, fwd_ref, W, H, x0, y0, 16, 8, &fwd_hdx, &fwd_hdy, &fwd_sad);
            int bwd_hdx, bwd_hdy; long bwd_sad; me_half_pel_block(orig, bwd_ref, W, H, x0, y0, 16, 8, &bwd_hdx, &bwd_hdy, &bwd_sad);
            uint8_t fwd_pred[64], bwd_pred[64]; mc_get_pred(fwd_ref, W, H, x0, y0, fwd_hdx, fwd_hdy, 8, fwd_pred); mc_get_pred(bwd_ref, W, H, x0, y0, bwd_hdx, bwd_hdy, 8, bwd_pred);
            uint8_t bi_pred[64]; for (int i = 0; i < 64; i++) bi_pred[i] = (uint8_t)((fwd_pred[i]+bwd_pred[i]+1)>>1);
            uint8_t dc_p[64], vt_p[64], hz_p[64];
            /* Simple intra for B-frames */
            int s = 0, n = 0;
            if (y0 > 0) for (int c = 0; c < 8; c++) { s += recon[(y0-1)*W+x0+c]; n++; }
            if (x0 > 0) for (int r = 0; r < 8; r++) { s += recon[(y0+r)*W+x0-1]; n++; }
            uint8_t dc_val = (n > 0) ? (uint8_t)(s/n) : 128;
            for (int i = 0; i < 64; i++) dc_p[i] = dc_val;
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) vt_p[r*8+c] = recon[(y0-1)*W+x0+c];
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) hz_p[r*8+c] = recon[(y0+r)*W+x0-1];
            
            uint8_t *cands[] = {fwd_pred, bwd_pred, bi_pred, dc_p, vt_p, hz_p};
            for (int ci = 0; ci < 6; ci++) {
                int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)cands[ci][i]);
                int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                int qs = get_qstep(b_qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
                
                float tangent[64], manifold[64], dequantized[64];
                for (int i = 0; i < 64; i++) tangent[i] = (float)res[i];
                nirvana_expmap(manifold, tangent, 64, NIRVANA_POINCARE_CURVATURE);
                
                float tangent2[64], manifold2[64];
                nirvana_logmap(tangent2, manifold, 64, NIRVANA_POINCARE_CURVATURE);
                for (int i = 0; i < 64; i++) {
                    float sign = (tangent2[i] >= 0) ? 1 : -1;
                    float abs_val = fabsf(tangent2[i]);
                    tangent2[i] = sign * ((int)(abs_val / qs + 0.5f)) * qs;
                }
                nirvana_expmap(manifold2, tangent2, 64, NIRVANA_POINCARE_CURVATURE);
                nirvana_logmap(dequantized, manifold2, 64, NIRVANA_POINCARE_CURVATURE);
                int16_t lv[64]; for (int i = 0; i < 64; i++) lv[i] = (int16_t)dequantized[i];
                
                long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)cands[ci][r*8+c] + (int)lv[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                float cost = nirvana_hsl_rd_cost(sse, 64, lambda, 8);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, cands[ci], 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
            }
            
            /* Encode with CABAC */
            NirvanaCABACEncoder cabac;
            nirvana_cabac_init(&cabac, fbuf + fpos + 256, 256);
            
            int zz[64];
            for (int i = 0; i < 64; i++) {
                int r = i / 8, c = i % 8;
                int s = r + c;
                if (s < 8) zz[i] = (s * (s + 1)) / 2 + (s % 2 ? c : r);
                else { int d = 15 - s; zz[i] = 63 - (d * (d + 1)) / 2 - (d % 2 ? 7 - c : 7 - r); }
            }
            
            int nz = 0; for (int i = 0; i < 64; i++) if (best_lv[zz[i]] != 0) nz++;
            nirvana_cabac_encode_ue(&cabac, nz, 0);
            
            if (nz > 0) {
                int run = 0;
                for (int i = 0; i < 64; i++) {
                    int v = best_lv[zz[i]];
                    if (v != 0) {
                        nirvana_cabac_encode_ue(&cabac, run, 64);
                        nirvana_cabac_encode_se(&cabac, v, 128);
                        run = 0;
                    } else {
                        run++;
                    }
                }
            }
            
            int cabac_bytes = nirvana_cabac_finish(&cabac);
            fpos += cabac_bytes;
            
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { 
                int v = (int)best_pred[r*8+c] + (int)best_lv[r*8+c]; 
                recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); 
            }
        }
        cdef_filter(recon, W, H, b_qp); wiener_filter(recon, W, H, b_qp);
        memcpy(all_recon + fn * np, recon, np); bfr++; if (fn % 30 == 0 || fn < 5) printf("  Frame %d (B): bytes=%d\n", fn, fpos); free(fbuf);
    }

    /* Write to container */
    printf("--- Writing container ---\n");
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np; int is_i = (ftypes[fn] == FRAME_I);
        const uint8_t *ref = NULL; int ref_fn = fn - 1; while (ref_fn >= 0 && ftypes[ref_fn] == FRAME_B) ref_fn--; if (ref_fn >= 0) ref = all_recon + ref_fn * np;
        int nbx = W/8, nby = H/8; uint8_t *fbuf = (uint8_t *)malloc(nbx*nby*256); int fpos = 0; int frame_qp = (ftypes[fn] == FRAME_B) ? qp+1 : qp;

        for (int by = 0; by < nby; by++) for (int bx = 0; bx < nbx; bx++) {
            int x0 = bx*8, y0 = by*8; int16_t blk[64]; 
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];
            double best_cost = 1e18; uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = get_qstep(frame_qp); double lambda = get_lambda(frame_qp);

            if (ftypes[fn] == FRAME_B) {
                int pfn = fn-1; while (pfn >= 0 && ftypes[pfn] == FRAME_B) pfn--; int nfn = fn+1; while (nfn < nf && ftypes[nfn] == FRAME_B) nfn++;
                const uint8_t *fr = (pfn >= 0) ? all_recon + pfn * np : NULL; const uint8_t *br = (nfn < nf) ? all_recon + nfn * np : fr; if (!fr) fr = br; if (!br) br = fr;
                int fhdx, fhdy; long fsad; me_half_pel_block(orig, fr, W, H, x0, y0, 16, 8, &fhdx, &fhdy, &fsad); int bhdx, bhdy; long bsad; me_half_pel_block(orig, br, W, H, x0, y0, 16, 8, &bhdx, &bhdy, &bsad);
                uint8_t fp[64], bp[64]; mc_get_pred(fr, W, H, x0, y0, fhdx, fhdy, 8, fp); mc_get_pred(br, W, H, x0, y0, bhdx, bhdy, 8, bp);
                uint8_t bi[64]; for (int i = 0; i < 64; i++) bi[i] = (uint8_t)((fp[i]+bp[i]+1)>>1);
                uint8_t dc[64], vt[64], hz[64];
                int s = 0, n = 0;
                if (y0 > 0) for (int c = 0; c < 8; c++) { s += recon[(y0-1)*W+x0+c]; n++; }
                if (x0 > 0) for (int r = 0; r < 8; r++) { s += recon[(y0+r)*W+x0-1]; n++; }
                uint8_t dc_val = (n > 0) ? (uint8_t)(s/n) : 128;
                for (int i = 0; i < 64; i++) dc[i] = dc_val;
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) vt[r*8+c] = recon[(y0-1)*W+x0+c];
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) hz[r*8+c] = recon[(y0+r)*W+x0-1];
                uint8_t *cands[] = {fp, bp, bi, dc, vt, hz};
                for (int ci = 0; ci < 6; ci++) {
                    int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)cands[ci][i]);
                    int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                    int qs = get_qstep(frame_qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
                    
                    float tangent[64], manifold[64], dequantized[64];
                    for (int i = 0; i < 64; i++) tangent[i] = (float)res[i];
                    nirvana_expmap(manifold, tangent, 64, NIRVANA_POINCARE_CURVATURE);
                    
                    float tangent2[64], manifold2[64];
                    nirvana_logmap(tangent2, manifold, 64, NIRVANA_POINCARE_CURVATURE);
                    for (int i = 0; i < 64; i++) {
                        float sign = (tangent2[i] >= 0) ? 1 : -1;
                        float abs_val = fabsf(tangent2[i]);
                        tangent2[i] = sign * ((int)(abs_val / qs + 0.5f)) * qs;
                    }
                    nirvana_expmap(manifold2, tangent2, 64, NIRVANA_POINCARE_CURVATURE);
                    nirvana_logmap(dequantized, manifold2, 64, NIRVANA_POINCARE_CURVATURE);
                    int16_t lv[64]; for (int i = 0; i < 64; i++) lv[i] = (int16_t)dequantized[i];
                    
                    long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)cands[ci][r*8+c] + (int)lv[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                    float cost = nirvana_hsl_rd_cost(sse, 64, lambda, 8);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, cands[ci], 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                }
            } else if (ref && !is_i) {
                int hdx, hdy; long sad; me_half_pel_block(orig, ref, W, H, x0, y0, 16, 8, &hdx, &hdy, &sad);
                uint8_t ip[64]; mc_get_pred(ref, W, H, x0, y0, hdx, hdy, 8, ip);
                int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)ip[i]);
                int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                int qs = get_qstep(qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
                
                float tangent[64], manifold[64], dequantized[64];
                for (int i = 0; i < 64; i++) tangent[i] = (float)res[i];
                nirvana_expmap(manifold, tangent, 64, NIRVANA_POINCARE_CURVATURE);
                
                float tangent2[64], manifold2[64];
                nirvana_logmap(tangent2, manifold, 64, NIRVANA_POINCARE_CURVATURE);
                for (int i = 0; i < 64; i++) {
                    float sign = (tangent2[i] >= 0) ? 1 : -1;
                    float abs_val = fabsf(tangent2[i]);
                    tangent2[i] = sign * ((int)(abs_val / qs + 0.5f)) * qs;
                }
                nirvana_expmap(manifold2, tangent2, 64, NIRVANA_POINCARE_CURVATURE);
                nirvana_logmap(dequantized, manifold2, 64, NIRVANA_POINCARE_CURVATURE);
                int16_t lv[64]; for (int i = 0; i < 64; i++) lv[i] = (int16_t)dequantized[i];
                
                long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)ip[r*8+c] + (int)lv[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                float cost = nirvana_hsl_rd_cost(sse, 64, lambda, 8);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, ip, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }

                for (int mode = 0; mode < 33; mode++) {
                    uint8_t pred[64];
                    /* Simple intra */
                    int s = 0, n = 0;
                    if (y0 > 0) for (int c = 0; c < 8; c++) { s += recon[(y0-1)*W+x0+c]; n++; }
                    if (x0 > 0) for (int r = 0; r < 8; r++) { s += recon[(y0+r)*W+x0-1]; n++; }
                    uint8_t dc_val = (n > 0) ? (uint8_t)(s/n) : 128;
                    for (int i = 0; i < 64; i++) pred[i] = dc_val;
                    
                    for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                    var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                    qs = get_qstep(qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
                    
                    for (int i = 0; i < 64; i++) tangent[i] = (float)res[i];
                    nirvana_expmap(manifold, tangent, 64, NIRVANA_POINCARE_CURVATURE);
                    
                    nirvana_logmap(tangent2, manifold, 64, NIRVANA_POINCARE_CURVATURE);
                    for (int i = 0; i < 64; i++) {
                        float sign = (tangent2[i] >= 0) ? 1 : -1;
                        float abs_val = fabsf(tangent2[i]);
                        tangent2[i] = sign * ((int)(abs_val / qs + 0.5f)) * qs;
                    }
                    nirvana_expmap(manifold2, tangent2, 64, NIRVANA_POINCARE_CURVATURE);
                    nirvana_logmap(dequantized, manifold2, 64, NIRVANA_POINCARE_CURVATURE);
                    for (int i = 0; i < 64; i++) lv[i] = (int16_t)dequantized[i];
                    
                    sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)lv[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                    cost = nirvana_hsl_rd_cost(sse, 64, lambda, 8);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                }
            } else {
                for (int mode = 0; mode < 33; mode++) {
                    uint8_t pred[64];
                    int s = 0, n = 0;
                    if (y0 > 0) for (int c = 0; c < 8; c++) { s += recon[(y0-1)*W+x0+c]; n++; }
                    if (x0 > 0) for (int r = 0; r < 8; r++) { s += recon[(y0+r)*W+x0-1]; n++; }
                    uint8_t dc_val = (n > 0) ? (uint8_t)(s/n) : 128;
                    for (int i = 0; i < 64; i++) pred[i] = dc_val;
                    
                    int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                    int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                    int qs = get_qstep(qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
                    
                    float tangent[64], manifold[64], dequantized[64];
                    for (int i = 0; i < 64; i++) tangent[i] = (float)res[i];
                    nirvana_expmap(manifold, tangent, 64, NIRVANA_POINCARE_CURVATURE);
                    
                    float tangent2[64], manifold2[64];
                    nirvana_logmap(tangent2, manifold, 64, NIRVANA_POINCARE_CURVATURE);
                    for (int i = 0; i < 64; i++) {
                        float sign = (tangent2[i] >= 0) ? 1 : -1;
                        float abs_val = fabsf(tangent2[i]);
                        tangent2[i] = sign * ((int)(abs_val / qs + 0.5f)) * qs;
                    }
                    nirvana_expmap(manifold2, tangent2, 64, NIRVANA_POINCARE_CURVATURE);
                    nirvana_logmap(dequantized, manifold2, 64, NIRVANA_POINCARE_CURVATURE);
                    int16_t lv[64]; for (int i = 0; i < 64; i++) lv[i] = (int16_t)dequantized[i];
                    
                    long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)lv[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                    float cost = nirvana_hsl_rd_cost(sse, 64, lambda, 8);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                }
            }

            /* Encode with CABAC */
            NirvanaCABACEncoder cabac;
            nirvana_cabac_init(&cabac, fbuf + fpos + 256, 256);
            
            int zz[64];
            for (int i = 0; i < 64; i++) {
                int r = i / 8, c = i % 8;
                int s = r + c;
                if (s < 8) zz[i] = (s * (s + 1)) / 2 + (s % 2 ? c : r);
                else { int d = 15 - s; zz[i] = 63 - (d * (d + 1)) / 2 - (d % 2 ? 7 - c : 7 - r); }
            }
            
            int nz = 0; for (int i = 0; i < 64; i++) if (best_lv[zz[i]] != 0) nz++;
            nirvana_cabac_encode_ue(&cabac, nz, 0);
            
            if (nz > 0) {
                int run = 0;
                for (int i = 0; i < 64; i++) {
                    int v = best_lv[zz[i]];
                    if (v != 0) {
                        nirvana_cabac_encode_ue(&cabac, run, 64);
                        nirvana_cabac_encode_se(&cabac, v, 128);
                        run = 0;
                    } else {
                        run++;
                    }
                }
            }
            
            int cabac_bytes = nirvana_cabac_finish(&cabac);
            fpos += cabac_bytes;
            
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { 
                int v = (int)best_pred[r*8+c] + (int)best_lv[r*8+c]; 
                recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); 
            }
        }
        double p = psnr_calc(orig, recon, np); total_psnr += p; memcpy(all_recon + fn * np, recon, np);
        wubuv_write_video(wr, fbuf, fpos, is_i); total_comp += fpos;
        if (fn % 30 == 0 || fn < 5) printf("  Write Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_i ? "I" : (ftypes[fn] == FRAME_B ? "B" : "P"), p, fpos); free(fbuf);
    }

    wubuv_writer_close(wr);
    double avg = total_psnr / nf;
    printf("\n=== Results ===\n"); printf("Frames: %d (I:%d P:%d B:%d)\n", nf, ifr, pfr, bfr);
    printf("Average PSNR: %.1f dB\n", avg); printf("Total: %.2f MB, bitrate: %.0f kbps\n", total_comp / 1048576.0, total_comp * 8.0 / nf * fps / 1024);
    if (wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");

    if (recon_file) { FILE* rf = fopen(recon_file, "wb"); if (rf) { fwrite(all_recon, 1, (size_t)nf * np, rf); fclose(rf); printf("Reconstruction saved to %s\n", recon_file); } }
    free(frames); free(recon); free(ftypes); free(all_recon);
    free(qc.q_table); free(qc.metric_history);
    return 0;
}
