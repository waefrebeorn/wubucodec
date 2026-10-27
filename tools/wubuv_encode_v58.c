/* wubuv_encode_v58.c — WuBuCodec v58: Per-Block Lambda Adjustment
 *
 * Based on x264's AQ (Adaptive Quantization) and mb-tree:
 * 1. Compute complexity per block (variance)
 * 2. Adjust lambda per block based on complexity
 * 3. Allocate more bits to complex blocks, fewer to simple blocks
 * 4. This maximizes overall quality for a given bitrate
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

static int get_qstep(int qp) {
    static const int t[52] = {
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722
    };
    if (qp < 0) qp = 0; if (qp > 51) qp = 51;
    return t[qp];
}

static double get_lambda(int qp) { return 0.85 * pow(2.0, (qp - 12.0) / 3.0); }

static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep/2) / qstep);
}

static inline int dequantize(int level, int qstep) { return level * qstep; }

static double rd_cost(long sse, int bits, double lambda) { return (double)sse + lambda * (double)bits; }

static float dct8_basis[8][8][8][8];
static int dct_ready = 0;

static void init_dct_basis(void) {
    if (dct_ready) return;
    for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) {
        float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
        float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
        for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++)
            dct8_basis[u][v][x][y] = cu * cv *
                cosf((2*x+1)*u*3.14159265f/16.0f) *
                cosf((2*y+1)*v*3.14159265f/16.0f);
    }
    dct_ready = 1;
}

static void dct8_forward(const int16_t *in, int16_t *out) {
    init_dct_basis();
    for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) {
        float sum = 0;
        for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++)
            sum += in[x*8+y] * dct8_basis[u][v][x][y];
        out[u*8+v] = (int16_t)roundf(sum);
    }
}

static void dct8_inverse(const int16_t *in, int16_t *out) {
    init_dct_basis();
    for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++) {
        float sum = 0;
        for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++)
            sum += in[u*8+v] * dct8_basis[u][v][x][y];
        out[x*8+y] = (int16_t)roundf(sum);
    }
}

static void adst8_forward(const int16_t *in, int16_t *out) {
    for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) {
        float sum = 0;
        for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++)
            sum += in[x*8+y] * sinf((2*x+1)*(u+1)*3.14159265f/17.0f) * sinf((2*y+1)*(v+1)*3.14159265f/17.0f);
        out[u*8+v] = (int16_t)roundf(sum * 0.25f);
    }
}

static void adst8_inverse(const int16_t *in, int16_t *out) {
    for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++) {
        float sum = 0;
        for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++)
            sum += in[u*8+v] * sinf((2*x+1)*(u+1)*3.14159265f/17.0f) * sinf((2*y+1)*(v+1)*3.14159265f/17.0f);
        out[x*8+y] = (int16_t)roundf(sum * 0.25f);
    }
}

typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;
typedef struct { int gop_size; int b_frames; } GOPConfig;

static FrameType get_gop_type(int fn, GOPConfig *gop) {
    if (fn % gop->gop_size == 0) return FRAME_I;
    if (gop->b_frames > 0 && (fn % gop->gop_size) % (gop->b_frames + 1) != 0) return FRAME_B;
    return FRAME_P;
}

static void intra_predict(uint8_t *rec, int W, int H, int bx, int by, int size, int mode, uint8_t *pred) {
    if (mode == 0) {
        int s = 0, n = 0;
        if (by > 0) for (int c = 0; c < size; c++) { int x = bx+c; if (x < W) { s += rec[(by-1)*W+x]; n++; } }
        if (bx > 0) for (int r = 0; r < size; r++) { int y = by+r; if (y < H) { s += rec[y*W+bx-1]; n++; } }
        uint8_t dc = (n > 0) ? (uint8_t)(s/n) : 128;
        for (int i = 0; i < size*size; i++) pred[i] = dc;
    } else if (mode == 1) {
        if (by == 0) { for (int i = 0; i < size*size; i++) pred[i] = 128; return; }
        for (int c = 0; c < size; c++) { int x = bx+c; if (x >= W) break; uint8_t v = rec[(by-1)*W+x]; for (int r = 0; r < size; r++) pred[r*size+c] = v; }
    } else if (mode == 2) {
        if (bx == 0) { for (int i = 0; i < size*size; i++) pred[i] = 128; return; }
        for (int r = 0; r < size; r++) { int y = by+r; if (y >= H) break; uint8_t v = rec[y*W+bx-1]; for (int c = 0; c < size; c++) pred[r*size+c] = v; }
    } else if (mode == 3) {
        for (int r = 0; r < size; r++) for (int c = 0; c < size; c++) {
            int v;
            if (by > 0 && bx > 0) {
                int top = rec[(by-1)*W+bx+c], left = rec[(by+r)*W+bx-1];
                int tr = (bx+c+1 < W) ? rec[(by-1)*W+bx+c+1] : top;
                int bot = (by+r < H) ? rec[(by+r)*W+bx-1] : left;
                v = ((size-1)*top + (size-1)*left + tr + bot + size) / (2*size);
            } else if (by > 0) v = rec[(by-1)*W+bx+c];
            else if (bx > 0) v = rec[(by+r)*W+bx-1];
            else v = 128;
            pred[r*size+c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    } else {
        int angle = (mode - 5) * 11;
        float angle_rad = angle * 3.14159265f / 180.0f;
        float dx = cosf(angle_rad), dy = sinf(angle_rad);
        for (int r = 0; r < size; r++) for (int c = 0; c < size; c++) {
            float fx = c + 0.5f - size/2.0f, fy = r + 0.5f - size/2.0f;
            float proj = fx * dx + fy * dy;
            int ref_x = bx + (int)(proj * dx + size/2.0f);
            int ref_y = by + (int)(proj * dy + size/2.0f);
            ref_x = ref_x < 0 ? 0 : (ref_x >= W ? W-1 : ref_x);
            ref_y = ref_y < 0 ? 0 : (ref_y >= H ? H-1 : ref_y);
            pred[r*size+c] = rec[ref_y*W+ref_x];
        }
    }
}

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
    printf("WuBuCodec v58 — Per-Block Lambda Adjustment (QP=%d)\n", qp);
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
            double best_cost = 1e18; uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = get_qstep(frame_qp); double lambda = get_lambda(frame_qp);
            int best_transform = 0;

            for (int ref_i = 0; ref_i < ref_count; ref_i++) {
                int hdx, hdy; long sad;
                me_quarter_pel_block(orig, ref_frames[ref_i], W, H, x0, y0, 16, 8, &hdx, &hdy, &sad);
                uint8_t pred[64]; mc_get_pred(ref_frames[ref_i], W, H, x0, y0, hdx, hdy, 8, pred);
                int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                
                /* Per-block lambda adjustment: complex blocks get lower lambda (more bits) */
                double lambda_adj = lambda;
                if (var > 5000) lambda_adj = lambda * 0.5; /* complex: spend more bits */
                else if (var > 2000) lambda_adj = lambda * 0.7;
                else if (var > 500) lambda_adj = lambda * 0.9;
                else if (var > 100) lambda_adj = lambda * 1.0;
                else if (var > 50) lambda_adj = lambda * 1.2; /* simple: spend fewer bits */
                else lambda_adj = lambda * 1.5;
                
                int qp_adj = 0;
                if (var > 10000) qp_adj = -10;
                else if (var > 5000) qp_adj = -8;
                else if (var > 2000) qp_adj = -5;
                else if (var > 500) qp_adj = -2;
                else if (var > 100) qp_adj = 0;
                else if (var > 50) qp_adj = 3;
                else qp_adj = 6;
                
                int adj_qp = frame_qp + qp_adj;
                if (adj_qp < 0) adj_qp = 0;
                if (adj_qp > 51) adj_qp = 51;
                int qs = get_qstep(adj_qp);
                
                int16_t coeff[64], lv[64], dq[64], rres[64];
                dct8_forward(res, coeff);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                dct8_inverse(dq, rres);
                long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                double cost = rd_cost(sse, 64, lambda_adj);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 0; }
                
                adst8_forward(res, coeff);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                adst8_inverse(dq, rres);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda_adj);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 1; }
                
                for (int i = 0; i < 64; i++) lv[i] = quantize(res[i], qs);
                for (int i = 0; i < 64; i++) rres[i] = (int16_t)dequantize(lv[i], qs);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda_adj);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 2; }
            }

            /* Intra prediction */
            int intra_modes[] = {0, 1, 2, 3, 5, 10, 16, 22};
            for (int mi = 0; mi < 8; mi++) {
                uint8_t pred[64]; intra_predict(recon, W, H, x0, y0, 8, intra_modes[mi], pred);
                int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                
                double lambda_adj = lambda;
                if (var > 5000) lambda_adj = lambda * 0.5;
                else if (var > 2000) lambda_adj = lambda * 0.7;
                else if (var > 500) lambda_adj = lambda * 0.9;
                else if (var > 100) lambda_adj = lambda * 1.0;
                else if (var > 50) lambda_adj = lambda * 1.2;
                else lambda_adj = lambda * 1.5;
                
                int qp_adj = 0;
                if (var > 10000) qp_adj = -10;
                else if (var > 5000) qp_adj = -8;
                else if (var > 2000) qp_adj = -5;
                else if (var > 500) qp_adj = -2;
                else if (var > 100) qp_adj = 0;
                else if (var > 50) qp_adj = 3;
                else qp_adj = 6;
                
                int adj_qp = frame_qp + qp_adj;
                if (adj_qp < 0) adj_qp = 0;
                if (adj_qp > 51) adj_qp = 51;
                int qs = get_qstep(adj_qp);
                
                int16_t coeff[64], lv[64], dq[64], rres[64];
                dct8_forward(res, coeff);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                dct8_inverse(dq, rres);
                long sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                double cost = rd_cost(sse, 64, lambda_adj);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 0; }
                
                adst8_forward(res, coeff);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                adst8_inverse(dq, rres);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda_adj);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 1; }
                
                for (int i = 0; i < 64; i++) lv[i] = quantize(res[i], qs);
                for (int i = 0; i < 64; i++) rres[i] = (int16_t)dequantize(lv[i], qs);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda_adj);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 2; }
            }

            int16_t dq[64], rres[64];
            if (best_transform == 0) {
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
                dct8_inverse(dq, rres);
            } else if (best_transform == 1) {
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
                adst8_inverse(dq, rres);
            } else {
                for (int i = 0; i < 64; i++) rres[i] = (int16_t)dequantize(best_lv[i], best_qs);
            }
            for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { 
                int v = (int)best_pred[r*8+c] + (int)rres[r*8+c]; 
                recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); 
            }
            
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
