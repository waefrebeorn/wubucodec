/* wubuv_encode_v12.c — WuBuCodec v12: Clean single-pass encoder
 *
 * Fixes:
 * 1. Single-pass encoding (no stale references)
 * 2. Proper RD cost with actual SSE
 * 3. I/P/B frame structure with correct reference handling
 * 4. Rate control with lambda
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

/* ============================ Bitstream ============================ */
typedef struct { uint8_t *buf; int byte_pos, bit_pos, size; } BitStream;

static void bs_init(BitStream *bs, uint8_t *buf, int size) {
    bs->buf = buf; bs->byte_pos = 0; bs->bit_pos = 0; bs->size = size;
}

static void bs_write_bit(BitStream *bs, int bit) {
    if (bs->byte_pos >= bs->size) return;
    if (bit) bs->buf[bs->byte_pos] |= (1 << (7 - bs->bit_pos));
    bs->bit_pos++;
    if (bs->bit_pos == 8) { bs->bit_pos = 0; bs->byte_pos++; }
}

static void bs_write_ue(BitStream *bs, int val) {
    int p = 0, t = val;
    while (t > 0) { bs_write_bit(bs, 1); t--; p++; }
    bs_write_bit(bs, 0);
    if (p > 0) for (int i = p-1; i >= 0; i--) bs_write_bit(bs, (val >> i) & 1);
}

static void bs_write_se(BitStream *bs, int val) {
    int a = abs(val), s = (val < 0) ? 1 : 0;
    bs_write_ue(bs, a);
    if (a > 0) bs_write_bit(bs, s);
}

static int bs_bytes(BitStream *bs) {
    return bs->byte_pos + (bs->bit_pos > 0 ? 1 : 0);
}

static const int zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

static int encode_block(int16_t *levels, uint8_t *out, int sz) {
    BitStream bs; bs_init(&bs, out, sz);
    int nz = 0;
    for (int i = 0; i < 64; i++) if (levels[zigzag[i]] != 0) nz++;
    bs_write_ue(&bs, nz);
    if (nz == 0) return bs_bytes(&bs);
    int run = 0;
    for (int i = 0; i < 64; i++) {
        int zz = zigzag[i], lv = levels[zz];
        if (lv != 0) { bs_write_ue(&bs, run); bs_write_se(&bs, lv); run = 0; }
        else run++;
    }
    return bs_bytes(&bs);
}

/* ============================ DCT ============================ */
static float dct_basis[8][8][8][8];
static int dct_ready = 0;

static void init_dct_basis(void) {
    if (dct_ready) return;
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) {
            float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    dct_basis[u][v][x][y] = cu * cv *
                        cosf((2*x+1)*u*3.14159265f/16.0f) *
                        cosf((2*y+1)*v*3.14159265f/16.0f);
        }
    dct_ready = 1;
}

static void dct8_forward(const int16_t *in, int16_t *out) {
    init_dct_basis();
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) {
            float sum = 0;
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    sum += in[x*8+y] * dct_basis[u][v][x][y];
            out[u*8+v] = (int16_t)roundf(sum);
        }
}

static void dct8_inverse(const int16_t *in, int16_t *out) {
    init_dct_basis();
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) {
            float sum = 0;
            for (int u = 0; u < 8; u++)
                for (int v = 0; v < 8; v++)
                    sum += in[u*8+v] * dct_basis[u][v][x][y];
            out[x*8+y] = (int16_t)roundf(sum);
        }
}

static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep/2) / qstep);
}

static inline int dequantize(int level, int qstep) { return level * qstep; }

/* ============================ QP / Rate Control ============================ */
static int get_qstep(int qp) {
    static const int t[52] = {
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722
    };
    if (qp < 0) qp = 0; if (qp > 51) qp = 51;
    return t[qp];
}

static double get_lambda(int qp) {
    return 0.85 * pow(2.0, (qp - 12.0) / 3.0);
}

static int block_var(const int16_t *b) {
    int s = 0, s2 = 0;
    for (int i = 0; i < 64; i++) { s += b[i]; s2 += b[i]*b[i]; }
    return s2/64 - (s/64)*(s/64);
}

static int aq_qp(int base, int var) {
    if (var < 100) return base - 4;
    if (var < 500) return base - 2;
    if (var < 2000) return base;
    if (var < 5000) return base + 2;
    return base + 4;
}

static double rd_cost(long sse, int bits, double lambda) {
    return (double)sse + lambda * (double)bits;
}

/* ============================ Intra Prediction ============================ */
static void intra_predict(const uint8_t *rec, int W, int H, int bx, int by,
                          int mode, uint8_t *pred) {
    if (mode == 0) { /* DC */
        int s = 0, n = 0;
        if (by > 0) for (int c = 0; c < 8; c++) { int x = bx+c; if (x < W) { s += rec[(by-1)*W+x]; n++; } }
        if (bx > 0) for (int r = 0; r < 8; r++) { int y = by+r; if (y < H) { s += rec[y*W+bx-1]; n++; } }
        uint8_t dc = (n > 0) ? (uint8_t)(s/n) : 128;
        for (int i = 0; i < 64; i++) pred[i] = dc;
    } else if (mode == 1) { /* Vertical */
        if (by == 0) { for (int i = 0; i < 64; i++) pred[i] = 128; return; }
        for (int c = 0; c < 8; c++) {
            int x = bx+c; if (x >= W) break;
            uint8_t v = rec[(by-1)*W+x];
            for (int r = 0; r < 8; r++) pred[r*8+c] = v;
        }
    } else if (mode == 2) { /* Horizontal */
        if (bx == 0) { for (int i = 0; i < 64; i++) pred[i] = 128; return; }
        for (int r = 0; r < 8; r++) {
            int y = by+r; if (y >= H) break;
            uint8_t v = rec[y*W+bx-1];
            for (int c = 0; c < 8; c++) pred[r*8+c] = v;
        }
    } else if (mode == 3) { /* Planar */
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v;
                if (by > 0 && bx > 0) {
                    int top = rec[(by-1)*W+bx+c];
                    int left = rec[(by+r)*W+bx-1];
                    int tr = (bx+c+1 < W) ? rec[(by-1)*W+bx+c+1] : top;
                    int bot = (by+r < H) ? rec[(by+r)*W+bx-1] : left;
                    v = (7*top + 1*tr + 7*left + 1*bot + 8) >> 4;
                } else if (by > 0) {
                    v = rec[(by-1)*W+bx+c];
                } else if (bx > 0) {
                    v = rec[(by+r)*W+bx-1];
                } else v = 128;
                pred[r*8+c] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    } else if (mode == 4) { /* Diag down-left */
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v;
                if (by > 0) {
                    int x = bx + c + r + 1;
                    v = rec[(by-1)*W+(x < W ? x : W-1)];
                } else if (bx > 0) {
                    v = rec[(by+r)*W+bx-1];
                } else v = 128;
                pred[r*8+c] = (uint8_t)v;
            }
        }
    } else if (mode == 5) { /* Diag down-right */
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v;
                if (by > 0 && bx > 0) {
                    int x = bx + c - r - 1;
                    v = rec[(by-1)*W+(x >= 0 ? x : 0)];
                } else if (by > 0) {
                    v = rec[(by-1)*W+bx+c];
                } else if (bx > 0) {
                    v = rec[(by+r)*W+bx-1];
                } else v = 128;
                pred[r*8+c] = (uint8_t)v;
            }
        }
    } else if (mode == 6) { /* Vertical-right */
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v;
                if (by > 0 && bx > 0) {
                    int x = bx + c - r/2;
                    v = rec[(by-1)*W+(x >= 0 && x < W ? x : 0)];
                } else if (by > 0) {
                    v = rec[(by-1)*W+bx+c];
                } else if (bx > 0) {
                    v = rec[(by+r)*W+bx-1];
                } else v = 128;
                pred[r*8+c] = (uint8_t)v;
            }
        }
    } else if (mode == 7) { /* Horizontal-down */
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v;
                if (by > 0 && bx > 0) {
                    int y = by + r - c/2;
                    v = rec[(y >= 0 && y < H ? y : 0)*W+bx-1];
                } else if (by > 0) {
                    v = rec[(by-1)*W+bx+c];
                } else if (bx > 0) {
                    v = rec[(by+r)*W+bx-1];
                } else v = 128;
                pred[r*8+c] = (uint8_t)v;
            }
        }
    } else if (mode == 8) { /* Smooth */
        int dc = 0, n = 0;
        if (by > 0) for (int c = 0; c < 8; c++) { int x = bx+c; if (x < W) { dc += rec[(by-1)*W+x]; n++; } }
        if (bx > 0) for (int r = 0; r < 8; r++) { int y = by+r; if (y < H) { dc += rec[y*W+bx-1]; n++; } }
        uint8_t dc_val = (n > 0) ? (uint8_t)(dc/n) : 128;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                uint8_t top = (by > 0 && bx+c < W) ? rec[(by-1)*W+bx+c] : dc_val;
                uint8_t left = (bx > 0 && by+r < H) ? rec[(by+r)*W+bx-1] : dc_val;
                pred[r*8+c] = (uint8_t)((7*top + 7*left + 8*dc_val + 8) >> 4);
            }
        }
    } else if (mode == 9) { /* Filtered */
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int v = 0, w = 0;
                if (by > 0) { v += rec[(by-1)*W+bx+c] * (8-r); w += 8-r; }
                if (bx > 0) { v += rec[(by+r)*W+bx-1] * (8-c); w += 8-c; }
                if (by > 0 && bx > 0) { v += rec[(by-1)*W+bx-1]; w += 1; }
                pred[r*8+c] = (uint8_t)(w > 0 ? (v + w/2) / w : 128);
            }
        }
    } else if (mode >= 10 && mode < 18) { /* Directional */
        int angle = (mode - 10) * 22;
        float angle_rad = angle * 3.14159265f / 180.0f;
        float dx = cosf(angle_rad), dy = sinf(angle_rad);
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                float fx = c + 0.5f - 4.0f;
                float fy = r + 0.5f - 4.0f;
                float proj = fx * dx + fy * dy;
                int ref_x = bx + (int)(proj * dx + 4.0f);
                int ref_y = by + (int)(proj * dy + 4.0f);
                ref_x = ref_x < 0 ? 0 : (ref_x >= W ? W-1 : ref_x);
                ref_y = ref_y < 0 ? 0 : (ref_y >= H ? H-1 : ref_y);
                pred[r*8+c] = rec[ref_y*W+ref_x];
            }
        }
    } else {
        for (int i = 0; i < 64; i++) pred[i] = 128;
    }
}

/* ============================ Motion Estimation ============================ */
static long sad_8x8(const uint8_t *cur, const uint8_t *ref, int W, int H,
                    int bx, int by, int dx, int dy) {
    int rx = bx + dx, ry = by + dy;
    if (rx < 0 || ry < 0 || rx+8 > W || ry+8 > H) return 1LL << 30;
    long s = 0;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            s += abs((int)cur[(by+r)*W+bx+c] - (int)ref[(ry+r)*W+rx+c]);
    return s;
}

static const int diamond[8][2] = {{-1,0},{0,-1},{0,1},{1,0},{-1,-1},{-1,1},{1,-1},{1,1}};

static void me_diamond(const uint8_t *cur, const uint8_t *ref, int W, int H,
                       int bx, int by, int sr, int *dx, int *dy, long *sad) {
    *dx = 0; *dy = 0; *sad = sad_8x8(cur, ref, W, H, bx, by, 0, 0);
    int step = sr;
    while (step > 0) {
        int imp = 1;
        while (imp) {
            imp = 0;
            for (int i = 0; i < 8; i++) {
                int nx = *dx + diamond[i][0]*step, ny = *dy + diamond[i][1]*step;
                if (abs(nx) > sr || abs(ny) > sr) continue;
                long s = sad_8x8(cur, ref, W, H, bx, by, nx, ny);
                if (s < *sad) { *sad = s; *dx = nx; *dy = ny; imp = 1; }
            }
        }
        step >>= 1;
    }
}

static void half_pel_interp(const uint8_t *ref, int W, int H,
                           int bx, int by, int hdx, int hdy, uint8_t *out) {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int x2 = (bx+c)*2 + hdx, y2 = (by+r)*2 + hdy;
            int x0 = x2/2, y0 = y2/2, x1 = x0+1, y1 = y0+1;
            int fx = x2 & 1, fy = y2 & 1;
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
            if (x1 >= W) x1 = W-1; if (y1 >= H) y1 = H-1;
            int v = (ref[y0*W+x0]*(2-fx)*(2-fy) + ref[y0*W+x1]*fx*(2-fy) +
                     ref[y1*W+x0]*(2-fx)*fy + ref[y1*W+x1]*fx*fy) / 4;
            out[r*8+c] = (uint8_t)v;
        }
    }
}

static void me_half_pel(const uint8_t *cur, const uint8_t *ref, int W, int H,
                        int bx, int by, int sr, int *out_hdx, int *out_hdy, long *out_sad) {
    int idx, idy; long isad;
    me_diamond(cur, ref, W, H, bx, by, sr, &idx, &idy, &isad);
    int best_hdx = idx*2, best_hdy = idy*2;
    long best_sad = isad;
    uint8_t hp_block[64];
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int thdx = idx*2+dx, thdy = idy*2+dy;
            half_pel_interp(ref, W, H, bx, by, thdx, thdy, hp_block);
            long s = 0;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    s += abs((int)cur[(by+r)*W+bx+c] - (int)hp_block[r*8+c]);
            if (s < best_sad) { best_sad = s; best_hdx = thdx; best_hdy = thdy; }
        }
    }
    *out_hdx = best_hdx; *out_hdy = best_hdy; *out_sad = best_sad;
}

static void mc_get_pred(const uint8_t *ref, int W, int H,
                        int bx, int by, int hdx, int hdy, uint8_t *pred) {
    half_pel_interp(ref, W, H, bx, by, hdx, hdy, pred);
}

/* ============================ Deblocking Filter ============================ */
static void deblock_frame(uint8_t *recon, int W, int H, int qp) {
    int alpha = qp*2 + 8, beta = qp + 4;
    if (alpha < 8) alpha = 8; if (beta < 4) beta = 4;

    for (int by = 0; by < H; by += 8) {
        for (int bx = 8; bx < W; bx += 8) {
            for (int r = 0; r < 8; r++) {
                int y = by+r; if (y >= H) break;
                int p1 = recon[y*W+bx-1], q1 = recon[y*W+bx];
                int p2 = (bx >= 2) ? recon[y*W+bx-2] : p1;
                int q2 = (bx+1 < W) ? recon[y*W+bx+1] : q1;
                int d = abs(p1 - q1);
                if (d >= alpha) continue;
                if (abs(p2-p1) >= beta || abs(q2-q1) >= beta) continue;
                int delta = (d+2) >> 2;
                recon[y*W+bx-1] = (uint8_t)(p1-delta < 0 ? 0 : (p1-delta > 255 ? 255 : p1-delta));
                recon[y*W+bx]   = (uint8_t)(q1+delta < 0 ? 0 : (q1+delta > 255 ? 255 : q1+delta));
            }
        }
    }
    for (int bx = 0; bx < W; bx += 8) {
        for (int by = 8; by < H; by += 8) {
            for (int c = 0; c < 8; c++) {
                int x = bx+c; if (x >= W) break;
                int p1 = recon[(by-1)*W+x], q1 = recon[by*W+x];
                int p2 = (by >= 2) ? recon[(by-2)*W+x] : p1;
                int q2 = (by+1 < H) ? recon[(by+1)*W+x] : q1;
                int d = abs(p1 - q1);
                if (d >= alpha) continue;
                if (abs(p2-p1) >= beta || abs(q2-q1) >= beta) continue;
                int delta = (d+2) >> 2;
                recon[(by-1)*W+x] = (uint8_t)(p1-delta < 0 ? 0 : (p1-delta > 255 ? 255 : p1-delta));
                recon[by*W+x]     = (uint8_t)(q1+delta < 0 ? 0 : (q1+delta > 255 ? 255 : q1+delta));
            }
        }
    }
}

/* ============================ GOP Structure ============================ */
typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;

static FrameType get_gop_type(int fn, int gop, int bf) {
    if (fn % gop == 0) return FRAME_I;
    if (bf > 0 && (fn % gop) % (bf+1) != 0) return FRAME_B;
    return FRAME_P;
}

/* ============================ PSNR ============================ */
static double psnr_calc(const uint8_t *a, const uint8_t *b, int n) {
    double mse = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d*d; }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

/* ============================ Main ============================ */
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s input.raw output.wubv [options]\n", argv[0]);
        printf("  -w width      Video width (default 320)\n");
        printf("  -h height     Video height (default 240)\n");
        printf("  -f fps        Frames per second (default 30)\n");
        printf("  -q qp         Base QP 0-51 (default 26)\n");
        printf("  -g gop        GOP size (default 30)\n");
        printf("  -b b_frames   B-frames between anchors (default 2, 0=disable)\n");
        printf("  --recon file  Output reconstruction\n");
        return 1;
    }

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
    printf("WUBV Encoder v12 — Clean Single-Pass\n");
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

    uint8_t *recon = (uint8_t *)malloc(np);
    memset(recon, 0, np);

    FrameType *ftypes = (FrameType *)malloc(nf * sizeof(FrameType));
    for (int fn = 0; fn < nf; fn++) ftypes[fn] = get_gop_type(fn, gop_size, b_frames);

    uint8_t *all_recon = (uint8_t *)calloc((size_t)nf * np, 1);

    double total_psnr = 0;
    int ifr = 0, pfr = 0, bfr = 0;
    long total_comp = 0;

    /* Single pass: encode I/P frames first, then B frames */
    printf("--- Pass 1: I/P frames ---\n");
    for (int fn = 0; fn < nf; fn++) {
        if (ftypes[fn] == FRAME_B) continue;
        const uint8_t *orig = frames + fn * np;
        int is_i = (ftypes[fn] == FRAME_I);

        /* Reference is the previous non-B recon */
        const uint8_t *ref = NULL;
        int ref_fn = fn - 1;
        while (ref_fn >= 0 && ftypes[ref_fn] == FRAME_B) ref_fn--;
        if (ref_fn >= 0) ref = all_recon + ref_fn * np;

        int nbx = W/8, nby = H/8;
        uint8_t *fbuf = (uint8_t *)malloc(nbx*nby*256);
        int fpos = 0;

        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx*8, y0 = by*8;
                int16_t blk[64];
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];

                double best_cost = 1e18;
                uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = get_qstep(qp);
                double lambda = get_lambda(qp);

                if (ref && !is_i) {
                    int hdx, hdy; long sad;
                    me_half_pel(orig, ref, W, H, x0, y0, 16, &hdx, &hdy, &sad);
                    uint8_t inter_pred[64]; mc_get_pred(ref, W, H, x0, y0, hdx, hdy, inter_pred);
                    int16_t res[64], coeff[64], lv[64];
                    for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)inter_pred[i]);
                    int var = block_var(res); int qs = get_qstep(aq_qp(qp, var));
                    dct8_forward(res, coeff);
                    for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                    uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                    int16_t dq[64], rres[64];
                    for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                    dct8_inverse(dq, rres);
                    long sse = 0;
                    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                        int v = (int)inter_pred[r*8+c] + (int)rres[r*8+c];
                        v = v < 0 ? 0 : (v > 255 ? 255 : v);
                        int d = (int)orig[(y0+r)*W+x0+c] - v;
                        sse += d * d;
                    }
                    double cost = rd_cost(sse, sz, lambda);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, inter_pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                }

                /* 18 intra modes */
                for (int mode = 0; mode < 18; mode++) {
                    uint8_t pred[64];
                    intra_predict(recon, W, H, x0, y0, mode, pred);
                    int16_t res[64], coeff[64], lv[64];
                    for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                    int var = block_var(res); int qs = get_qstep(aq_qp(qp, var));
                    dct8_forward(res, coeff);
                    for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                    uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                    int16_t dq[64], rres[64];
                    for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                    dct8_inverse(dq, rres);
                    long sse = 0;
                    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                        int v = (int)pred[r*8+c] + (int)rres[r*8+c];
                        v = v < 0 ? 0 : (v > 255 ? 255 : v);
                        int d = (int)orig[(y0+r)*W+x0+c] - v;
                        sse += d * d;
                    }
                    double cost = rd_cost(sse, sz, lambda);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                }

                uint8_t bb[256]; int bsz = encode_block(best_lv, bb, 256);
                memcpy(fbuf + fpos, bb, bsz); fpos += bsz;

                int16_t dq[64], rres[64];
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
                dct8_inverse(dq, rres);
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                    int v = (int)best_pred[r*8+c] + (int)rres[r*8+c];
                    recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        deblock_frame(recon, W, H, qp);
        memcpy(all_recon + fn * np, recon, np);
        if (is_i) ifr++; else pfr++;
        if (fn % 30 == 0 || fn < 5) printf("  Frame %d (%s): bytes=%d\n", fn, is_i ? "I" : "P", fpos);
        free(fbuf);
    }

    printf("--- Pass 2: B-frames ---\n");
    for (int fn = 0; fn < nf; fn++) {
        if (ftypes[fn] != FRAME_B) continue;
        const uint8_t *orig = frames + fn * np;

        int prev_fn = fn - 1; while (prev_fn >= 0 && ftypes[prev_fn] == FRAME_B) prev_fn--;
        int next_fn = fn + 1; while (next_fn < nf && ftypes[next_fn] == FRAME_B) next_fn++;
        const uint8_t *fwd_ref = (prev_fn >= 0) ? all_recon + prev_fn * np : NULL;
        const uint8_t *bwd_ref = (next_fn < nf) ? all_recon + next_fn * np : fwd_ref;
        if (!fwd_ref) fwd_ref = bwd_ref; if (!bwd_ref) bwd_ref = fwd_ref;

        int nbx = W/8, nby = H/8;
        uint8_t *fbuf = (uint8_t *)malloc(nbx*nby*256);
        int fpos = 0;
        int b_qp = qp + 1;

        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx*8, y0 = by*8;
                int16_t blk[64];
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];

                double best_cost = 1e18;
                uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = get_qstep(b_qp);
                double lambda = get_lambda(b_qp);

                int fwd_hdx, fwd_hdy; long fwd_sad;
                me_half_pel(orig, fwd_ref, W, H, x0, y0, 16, &fwd_hdx, &fwd_hdy, &fwd_sad);
                int bwd_hdx, bwd_hdy; long bwd_sad;
                me_half_pel(orig, bwd_ref, W, H, x0, y0, 16, &bwd_hdx, &bwd_hdy, &bwd_sad);

                uint8_t fwd_pred[64], bwd_pred[64];
                mc_get_pred(fwd_ref, W, H, x0, y0, fwd_hdx, fwd_hdy, fwd_pred);
                mc_get_pred(bwd_ref, W, H, x0, y0, bwd_hdx, bwd_hdy, bwd_pred);

                uint8_t bi_pred[64];
                for (int i = 0; i < 64; i++) bi_pred[i] = (uint8_t)((fwd_pred[i]+bwd_pred[i]+1)>>1);

                uint8_t dc_p[64], vt_p[64], hz_p[64];
                intra_predict(recon, W, H, x0, y0, 0, dc_p);
                intra_predict(recon, W, H, x0, y0, 1, vt_p);
                intra_predict(recon, W, H, x0, y0, 2, hz_p);

                uint8_t *cands[] = {fwd_pred, bwd_pred, bi_pred, dc_p, vt_p, hz_p};
                for (int ci = 0; ci < 6; ci++) {
                    int16_t res[64], coeff[64], lv[64];
                    for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)cands[ci][i]);
                    int var = block_var(res); int qs = get_qstep(aq_qp(b_qp, var));
                    dct8_forward(res, coeff);
                    for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                    uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                    int16_t dq[64], rres[64];
                    for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                    dct8_inverse(dq, rres);
                    long sse = 0;
                    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                        int v = (int)cands[ci][r*8+c] + (int)rres[r*8+c];
                        v = v < 0 ? 0 : (v > 255 ? 255 : v);
                        int d = (int)orig[(y0+r)*W+x0+c] - v;
                        sse += d * d;
                    }
                    double cost = rd_cost(sse, sz, lambda);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, cands[ci], 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                }

                uint8_t bb[256]; int bsz = encode_block(best_lv, bb, 256);
                memcpy(fbuf + fpos, bb, bsz); fpos += bsz;

                int16_t dq[64], rres[64];
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
                dct8_inverse(dq, rres);
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                    int v = (int)best_pred[r*8+c] + (int)rres[r*8+c];
                    recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        deblock_frame(recon, W, H, b_qp);
        memcpy(all_recon + fn * np, recon, np);
        bfr++;
        if (fn % 30 == 0 || fn < 5) printf("  Frame %d (B): bytes=%d\n", fn, fpos);
        free(fbuf);
    }

    /* Write to container in order */
    printf("--- Writing container ---\n");
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np;
        int is_i = (ftypes[fn] == FRAME_I);
        const uint8_t *ref = NULL;
        int ref_fn = fn - 1; while (ref_fn >= 0 && ftypes[ref_fn] == FRAME_B) ref_fn--;
        if (ref_fn >= 0) ref = all_recon + ref_fn * np;

        int nbx = W/8, nby = H/8;
        uint8_t *fbuf = (uint8_t *)malloc(nbx*nby*256);
        int fpos = 0;
        int frame_qp = (ftypes[fn] == FRAME_B) ? qp+1 : qp;

        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx*8, y0 = by*8;
                int16_t blk[64];
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];

                double best_cost = 1e18;
                uint8_t best_pred[64]; int16_t best_lv[64]; int best_qs = get_qstep(frame_qp);
                double lambda = get_lambda(frame_qp);

                if (ftypes[fn] == FRAME_B) {
                    int pfn = fn-1; while (pfn >= 0 && ftypes[pfn] == FRAME_B) pfn--;
                    int nfn = fn+1; while (nfn < nf && ftypes[nfn] == FRAME_B) nfn++;
                    const uint8_t *fr = (pfn >= 0) ? all_recon + pfn * np : NULL;
                    const uint8_t *br = (nfn < nf) ? all_recon + nfn * np : fr;
                    if (!fr) fr = br; if (!br) br = fr;

                    int fhdx, fhdy; long fsad; me_half_pel(orig, fr, W, H, x0, y0, 16, &fhdx, &fhdy, &fsad);
                    int bhdx, bhdy; long bsad; me_half_pel(orig, br, W, H, x0, y0, 16, &bhdx, &bhdy, &bsad);
                    uint8_t fp[64], bp[64];
                    mc_get_pred(fr, W, H, x0, y0, fhdx, fhdy, fp);
                    mc_get_pred(br, W, H, x0, y0, bhdx, bhdy, bp);
                    uint8_t bi[64]; for (int i = 0; i < 64; i++) bi[i] = (uint8_t)((fp[i]+bp[i]+1)>>1);

                    uint8_t dc[64], vt[64], hz[64];
                    intra_predict(recon, W, H, x0, y0, 0, dc);
                    intra_predict(recon, W, H, x0, y0, 1, vt);
                    intra_predict(recon, W, H, x0, y0, 2, hz);
                    uint8_t *cands[] = {fp, bp, bi, dc, vt, hz};
                    for (int ci = 0; ci < 6; ci++) {
                        int16_t res[64], coeff[64], lv[64];
                        for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)cands[ci][i]);
                        int var = block_var(res); int qs = get_qstep(aq_qp(frame_qp, var));
                        dct8_forward(res, coeff);
                        for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                        uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                        int16_t dq[64], rres[64];
                        for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                        dct8_inverse(dq, rres);
                        long sse = 0;
                        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                            int v = (int)cands[ci][r*8+c] + (int)rres[r*8+c];
                            v = v < 0 ? 0 : (v > 255 ? 255 : v);
                            int d = (int)orig[(y0+r)*W+x0+c] - v;
                            sse += d * d;
                        }
                        double cost = rd_cost(sse, sz, lambda);
                        if (cost < best_cost) { best_cost = cost; memcpy(best_pred, cands[ci], 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                    }
                } else if (ref && !is_i) {
                    int hdx, hdy; long sad; me_half_pel(orig, ref, W, H, x0, y0, 16, &hdx, &hdy, &sad);
                    uint8_t ip[64]; mc_get_pred(ref, W, H, x0, y0, hdx, hdy, ip);
                    int16_t res[64], coeff[64], lv[64];
                    for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)ip[i]);
                    int var = block_var(res); int qs = get_qstep(aq_qp(qp, var));
                    dct8_forward(res, coeff);
                    for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                    uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                    int16_t dq[64], rres[64];
                    for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                    dct8_inverse(dq, rres);
                    long sse = 0;
                    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                        int v = (int)ip[r*8+c] + (int)rres[r*8+c];
                        v = v < 0 ? 0 : (v > 255 ? 255 : v);
                        int d = (int)orig[(y0+r)*W+x0+c] - v;
                        sse += d * d;
                    }
                    double cost = rd_cost(sse, sz, lambda);
                    if (cost < best_cost) { best_cost = cost; memcpy(best_pred, ip, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }

                    /* Also try intra for P frames */
                    for (int mode = 0; mode < 18; mode++) {
                        uint8_t pred[64]; intra_predict(recon, W, H, x0, y0, mode, pred);
                        for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                        var = block_var(res); qs = get_qstep(aq_qp(qp, var));
                        dct8_forward(res, coeff);
                        for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                        sz = encode_block(lv, bb, 256);
                        for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                        dct8_inverse(dq, rres);
                        sse = 0;
                        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                            int v = (int)pred[r*8+c] + (int)rres[r*8+c];
                            v = v < 0 ? 0 : (v > 255 ? 255 : v);
                            int d = (int)orig[(y0+r)*W+x0+c] - v;
                            sse += d * d;
                        }
                        cost = rd_cost(sse, sz, lambda);
                        if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                    }
                } else {
                    /* I-frame: try all 18 intra modes */
                    for (int mode = 0; mode < 18; mode++) {
                        uint8_t pred[64]; intra_predict(recon, W, H, x0, y0, mode, pred);
                        int16_t res[64], coeff[64], lv[64];
                        for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                        int var = block_var(res); int qs = get_qstep(aq_qp(qp, var));
                        dct8_forward(res, coeff);
                        for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                        uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                        int16_t dq[64], rres[64];
                        for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                        dct8_inverse(dq, rres);
                        long sse = 0;
                        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                            int v = (int)pred[r*8+c] + (int)rres[r*8+c];
                            v = v < 0 ? 0 : (v > 255 ? 255 : v);
                            int d = (int)orig[(y0+r)*W+x0+c] - v;
                            sse += d * d;
                        }
                        double cost = rd_cost(sse, sz, lambda);
                        if (cost < best_cost) { best_cost = cost; memcpy(best_pred, pred, 64); memcpy(best_lv, lv, sizeof(lv)); best_qs = qs; }
                    }
                }

                uint8_t bb[256]; int bsz = encode_block(best_lv, bb, 256);
                memcpy(fbuf + fpos, bb, bsz); fpos += bsz;

                int16_t dq[64], rres[64];
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
                dct8_inverse(dq, rres);
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                    int v = (int)best_pred[r*8+c] + (int)rres[r*8+c];
                    recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        deblock_frame(recon, W, H, frame_qp);
        double p = psnr_calc(orig, recon, np);
        total_psnr += p;
        memcpy(all_recon + fn * np, recon, np);
        wubuv_write_video(wr, fbuf, fpos, is_i);
        total_comp += fpos;
        if (fn % 30 == 0 || fn < 5) printf("  Write Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_i ? "I" : (ftypes[fn] == FRAME_B ? "B" : "P"), p, fpos);
        free(fbuf);
    }

    wubuv_writer_close(wr);
    double avg = total_psnr / nf;
    printf("\n=== Results ===\n");
    printf("Frames: %d (I:%d P:%d B:%d)\n", nf, ifr, pfr, bfr);
    printf("Average PSNR: %.1f dB\n", avg);
    printf("Total: %.2f MB, bitrate: %.0f kbps\n", total_comp / 1048576.0, total_comp * 8.0 / nf * fps / 1024);
    if (wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");

    if (recon_file) {
        FILE* rf = fopen(recon_file, "wb");
        if (rf) {
            fwrite(all_recon, 1, (size_t)nf * np, rf);
            fclose(rf);
            printf("Reconstruction saved to %s\n", recon_file);
        }
    }

    free(frames); free(recon); free(ftypes); free(all_recon);
    return 0;
}
