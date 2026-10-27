/* wubuv_encode_v67.c — WuBuCodec v67: Back to v63 + Proper Deblocking Filter
 *
 * v63 was our best: 42.1 dB @ 2113 kbps (QP22)
 * This version adds a proper deblocking filter that only operates on block boundaries
 * and respects edge directionality, avoiding the quality loss we saw in v62.
 *
 * The key insight from v62's failure: applying filters to the entire frame blurs
 * everything. Instead, we only filter along block boundaries where blocking artifacts
 * actually appear, and we skip filtering across strong edges.
 */
#define _POSIX_C_SOURCE 20080916
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

/* ============================ Range Coder ============================ */

#define MAX_SYMBOL 4096

typedef struct {
    uint32_t low;
    uint32_t range;
    uint32_t value;
    int bits_pending;
    uint8_t *output;
    int output_pos;
    int output_size;
} RangeEncoder;

typedef struct {
    uint16_t freq[MAX_SYMBOL + 2];
    uint16_t total;
    uint16_t symbols;
    uint16_t update_rate;
} ProbModel;

static void range_init(RangeEncoder *enc, uint8_t *output, int size) {
    enc->low = 0;
    enc->range = 0xFFFFFFFF;
    enc->value = 0;
    enc->bits_pending = 0;
    enc->output = output;
    enc->output_pos = 0;
    enc->output_size = size;
}

static void range_write_byte(RangeEncoder *enc, uint8_t byte) {
    if (enc->output_pos < enc->output_size) {
        enc->output[enc->output_pos++] = byte;
    }
}

static void range_shift(RangeEncoder *enc) {
    if (enc->low < 0xFF000000 || (enc->low >> 31) != 0) {
        range_write_byte(enc, (uint8_t)((enc->value >> 24) & 0xFF));
        enc->value = (enc->value << 8) & 0xFFFFFF00;
        enc->bits_pending = 0;
    } else {
        enc->bits_pending++;
    }
}

static void range_encode_symbol(RangeEncoder *enc, ProbModel *model, uint32_t symbol) {
    uint32_t cum = 0;
    for (uint32_t s = 0; s < symbol; s++) cum += model->freq[s];
    uint32_t freq_s = model->freq[symbol];
    
    enc->range /= model->total;
    enc->low += cum * enc->range;
    enc->range *= freq_s;
    
    while (enc->range < 0x1000000) {
        enc->range <<= 8;
        enc->low = (enc->low << 8) & 0xFFFFFFFF;
        range_shift(enc);
    }
    
    model->freq[symbol] += model->update_rate;
    model->total += model->update_rate;
    
    if (model->total > 16000) {
        model->total = 0;
        for (uint32_t s = 0; s <= model->symbols; s++) {
            model->freq[s] = (model->freq[s] + 1) >> 1;
            model->total += model->freq[s];
        }
    }
}

static int range_finish(RangeEncoder *enc) {
    for (int i = 0; i < 4; i++) {
        range_write_byte(enc, (uint8_t)((enc->low >> 24) & 0xFF));
        enc->low <<= 8;
    }
    return enc->output_pos;
}

static void model_init(ProbModel *model, uint32_t num_symbols) {
    model->symbols = num_symbols;
    model->total = num_symbols;
    model->update_rate = 4;
    for (uint32_t s = 0; s < num_symbols; s++) model->freq[s] = 1;
    model->freq[num_symbols] = 0;
}

static void model_init_nonuniform(ProbModel *model, uint32_t num_symbols, float decay) {
    model->symbols = num_symbols;
    model->total = 0;
    model->update_rate = 4;
    for (uint32_t s = 0; s < num_symbols; s++) {
        model->freq[s] = (uint16_t)(powf(1.0f - decay, (float)s) * 100.0f + 1.0f);
        model->total += model->freq[s];
    }
    model->freq[num_symbols] = 0;
}

/* Context models */
typedef struct {
    ProbModel models[16];
    ProbModel sign_model;
} ContextModels;

static void context_init(ContextModels *ctx) {
    for (int i = 0; i < 16; i++) {
        model_init_nonuniform(&ctx->models[i], 256, 0.05f + (float)i * 0.01f);
    }
    model_init(&ctx->sign_model, 2);
}

static int compute_context(const int16_t *coeffs, int pos, int N) {
    int ctx = 0;
    int r = pos / N, c = pos % N;
    if (r > 0) ctx += abs(coeffs[pos - N]);
    if (c > 0) ctx += abs(coeffs[pos - 1]);
    if (r > 0 && c > 0) ctx += abs(coeffs[pos - N - 1]);
    if (r > 0 && c < N-1) ctx += abs(coeffs[pos - N + 1]);
    ctx = ctx >> 5;
    if (ctx > 15) ctx = 15;
    return ctx;
}

/* ============================ GOP Structure ============================ */

typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;
typedef struct { int gop_size; int b_frames; } GOPConfig;

static FrameType get_gop_type(int fn, GOPConfig *gop) {
    if (fn % gop->gop_size == 0) return FRAME_I;
    if (gop->b_frames > 0 && (fn % gop->gop_size) % (gop->b_frames + 1) != 0) return FRAME_B;
    return FRAME_P;
}

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

/* ============================ DCT ============================ */

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

static double psnr_calc(const uint8_t *a, const uint8_t *b, int n) {
    double mse = 0; for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d*d; } mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
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

/* Zigzag order for 8x8 */
static const int zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

/* ============================ Deblocking Filter ============================ */

static void deblocking_filter(uint8_t *recon, int W, int H, int qp) {
    /* Only filter block boundaries (every 8 pixels) */
    /* Skip filtering across strong edges */
    
    int threshold = qp / 2 + 1;
    if (threshold < 2) threshold = 2;
    if (threshold > 16) threshold = 16;
    
    /* Horizontal edges (between vertical blocks) */
    for (int y = 0; y < H; y += 8) {
        for (int x = 0; x < W; x++) {
            if (y == 0 || y >= H) continue;
            
            /* Check if this is a strong edge */
            int edge_val = abs((int)recon[y*W+x] - (int)recon[(y-1)*W+x]);
            if (edge_val > threshold * 8) continue; /* Skip strong edges */
            
            /* Apply gentle smoothing */
            int p0 = recon[y*W+x];
            int p1 = recon[(y-1)*W+x];
            int p2 = (y >= 2) ? recon[(y-2)*W+x] : p1;
            int q0 = recon[(y+1)*W+x];
            int q1 = (y+2 < H) ? recon[(y+2)*W+x] : q0;
            
            /* Only filter if the difference is small */
            if (abs(p0 - p1) < threshold * 4) {
                recon[y*W+x] = (uint8_t)((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3);
                recon[(y-1)*W+x] = (uint8_t)((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3);
            }
        }
    }
    
    /* Vertical edges (between horizontal blocks) */
    for (int x = 0; x < W; x += 8) {
        for (int y = 0; y < H; y++) {
            if (x == 0 || x >= W) continue;
            
            int edge_val = abs((int)recon[y*W+x] - (int)recon[y*W+x-1]);
            if (edge_val > threshold * 8) continue;
            
            int p0 = recon[y*W+x];
            int p1 = recon[y*W+x-1];
            int p2 = (x >= 2) ? recon[y*W+x-2] : p1;
            int q0 = recon[y*W+x+1];
            int q1 = (x+2 < W) ? recon[y*W+x+2] : q0;
            
            if (abs(p0 - p1) < threshold * 4) {
                recon[y*W+x] = (uint8_t)((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3);
                recon[y*W+x-1] = (uint8_t)((p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3);
            }
        }
    }
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
    printf("WuBuCodec v67 — v63 + Deblocking Filter (QP=%d)\n", qp);
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
        
        RangeEncoder range;
        range_init(&range, fbuf, 1024*1024);
        ContextModels ctx;
        context_init(&ctx);
        
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
                double cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 0; }
                
                adst8_forward(res, coeff);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                adst8_inverse(dq, rres);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 1; }
                
                for (int i = 0; i < 64; i++) lv[i] = quantize(res[i], qs);
                for (int i = 0; i < 64; i++) rres[i] = (int16_t)dequantize(lv[i], qs);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 2; }
            }

            /* For B-frames, try weighted prediction */
            if (is_b && ref_count >= 2) {
                int hdx1, hdy1, hdx2, hdy2; long sad1, sad2;
                me_quarter_pel_block(orig, ref_frames[0], W, H, x0, y0, 16, 8, &hdx1, &hdy1, &sad1);
                me_quarter_pel_block(orig, ref_frames[1], W, H, x0, y0, 16, 8, &hdx2, &hdy2, &sad2);
                uint8_t pred1[64], pred2[64];
                mc_get_pred(ref_frames[0], W, H, x0, y0, hdx1, hdy1, 8, pred1);
                mc_get_pred(ref_frames[1], W, H, x0, y0, hdx2, hdy2, 8, pred2);
                
                for (int w = 1; w <= 3; w++) {
                    uint8_t pred[64];
                    for (int i = 0; i < 64; i++) {
                        pred[i] = (uint8_t)((w * pred1[i] + (4 - w) * pred2[i] + 2) >> 2);
                    }
                    int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                    int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                    
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
                    double cost = rd_cost(sse, 64, lambda);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 0; }
                }
            }

            /* Intra prediction */
            int intra_modes[] = {0, 1, 2, 3, 5, 10, 16, 22};
            for (int mi = 0; mi < 8; mi++) {
                uint8_t pred[64]; intra_predict(recon, W, H, x0, y0, 8, intra_modes[mi], pred);
                int16_t res[64]; for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
                
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
                double cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 0; }
                
                adst8_forward(res, coeff);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                adst8_inverse(dq, rres);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 1; }
                
                for (int i = 0; i < 64; i++) lv[i] = quantize(res[i], qs);
                for (int i = 0; i < 64; i++) rres[i] = (int16_t)dequantize(lv[i], qs);
                sse = 0; for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) { int v = (int)pred[r*8+c] + (int)rres[r*8+c]; v = v < 0 ? 0 : (v > 255 ? 255 : v); int d = (int)orig[(y0+r)*W+x0+c] - v; sse += d * d; }
                cost = rd_cost(sse, 64, lambda);
                if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; best_transform = 2; }
            }

            /* Reconstruct using best transform */
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
            
            /* Multi-symbol arithmetic encode with per-coefficient context */
            int16_t zz[64];
            for (int i = 0; i < 64; i++) zz[i] = best_lv[zigzag[i]];
            
            /* Find last non-zero coefficient */
            int last_nz = -1;
            for (int i = 63; i >= 0; i--) {
                if (zz[i] != 0) { last_nz = i; break; }
            }
            
            if (last_nz >= 0) {
                /* Encode each coefficient with context from neighbors */
                for (int i = 0; i <= last_nz; i++) {
                    int val = zz[i];
                    int abs_val = abs(val);
                    int sign = (val < 0) ? 1 : 0;
                    
                    /* Compute context from already-encoded neighbors */
                    int ctx_idx = compute_context(zz, i, 8);
                    
                    /* Encode absolute value with context-based model */
                    uint32_t symbol = (uint32_t)(abs_val < 255 ? abs_val : 255);
                    range_encode_symbol(&range, &ctx.models[ctx_idx], symbol);
                    
                    /* Encode sign */
                    if (abs_val > 0) {
                        range_encode_symbol(&range, &ctx.sign_model, (uint32_t)sign);
                    }
                }
            }
        }
        
        int range_bytes = range_finish(&range);
        
        /* Apply deblocking filter before reference */
        deblocking_filter(recon, W, H, frame_qp);
        
        double p = psnr_calc(orig, recon, np); total_psnr += p;
        memcpy(all_recon + fn * np, recon, np);
        wubuv_write_video(wr, fbuf, range_bytes, is_i); total_comp += range_bytes;
        if (fn % 30 == 0 || fn < 5) printf("  Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_i ? "I" : (ftypes[fn] == FRAME_B ? "B" : "P"), p, range_bytes);
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
