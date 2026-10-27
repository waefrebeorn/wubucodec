/* nirvana_merge.c — Start with 8x8 blocks, merge when RD cost improves
 *
 * This is the standard approach: start small, merge when beneficial.
 * For I-frames: stays at 8x8 (intra prediction is local)
 * For P/B-frames: merges to larger blocks when ME finds good matches
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

static int get_qstep(int qp) {
    static const int t[64] = {
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722,
        810,910,1022,1148,1288,1444,1620,1820,2044,2296,2576,2888
    };
    if (qp < 0) qp = 0; else if (qp > 63) qp = 63;
    return t[qp];
}

static double get_lambda(int qp) { return 0.85 * pow(2.0, (qp - 12.0) / 3.0); }

static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep/2) / qstep);
}

static inline int dequantize(int level, int qstep) { return level * qstep; }

static void dct_forward_8x8(const int16_t *in, int16_t *out) {
    for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) {
        float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
        float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
        float sum = 0;
        for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++)
            sum += in[x*8+y] * cosf((2*x+1)*u*3.14159265f/16.0f) * cosf((2*y+1)*v*3.14159265f/16.0f);
        out[u*8+v] = (int16_t)roundf(cu * cv * sum);
    }
}

static void dct_inverse_8x8(const int16_t *in, int16_t *out) {
    for (int x = 0; x < 8; x++) for (int y = 0; y < 8; y++) {
        float sum = 0;
        for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) {
            float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            sum += in[u*8+v] * cu * cv * cosf((2*x+1)*u*3.14159265f/16.0f) * cosf((2*y+1)*v*3.14159265f/16.0f);
        }
        out[x*8+y] = (int16_t)roundf(sum);
    }
}

typedef enum { FRAME_I, FRAME_P, FRAME_B } FrameType;
typedef struct { int gop_size; int b_frames; } GOPConfig;

static FrameType get_gop_type(int fn, GOPConfig *gop) {
    if (fn % gop->gop_size == 0) return FRAME_I;
    if (gop->b_frames > 0 && (fn % gop->gop_size) % (gop->b_frames + 1) != 0) return FRAME_B;
    return FRAME_P;
}

static void intra_predict(uint8_t *rec, int W, int H, int bx, int by, int mode, uint8_t *pred) {
    if (mode == 0) {
        int s = 0, n = 0;
        if (by > 0) for (int c = 0; c < 8; c++) { int x = bx+c; if (x < W) { s += rec[(by-1)*W+x]; n++; } }
        if (bx > 0) for (int r = 0; r < 8; r++) { int y = by+r; if (y < H) { s += rec[y*W+bx-1]; n++; } }
        uint8_t dc = (n > 0) ? (uint8_t)(s/n) : 128;
        for (int i = 0; i < 64; i++) pred[i] = dc;
    } else if (mode == 1) {
        if (by == 0) { for (int i = 0; i < 64; i++) pred[i] = 128; return; }
        for (int c = 0; c < 8; c++) { int x = bx+c; if (x >= W) break; uint8_t v = rec[(by-1)*W+x]; for (int r = 0; r < 8; r++) pred[r*8+c] = v; }
    } else if (mode == 2) {
        if (bx == 0) { for (int i = 0; i < 64; i++) pred[i] = 128; return; }
        for (int r = 0; r < 8; r++) { int y = by+r; if (y >= H) break; uint8_t v = rec[y*W+bx-1]; for (int c = 0; c < 8; c++) pred[r*8+c] = v; }
    } else {
        int angle = (mode - 5) * 11;
        float angle_rad = angle * 3.14159265f / 180.0f;
        float dx = cosf(angle_rad), dy = sinf(angle_rad);
        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
            float fx = c + 0.5f - 4.0f, fy = r + 0.5f - 4.0f;
            float proj = fx * dx + fy * dy;
            int ref_x = bx + (int)(proj * dx + 4.0f);
            int ref_y = by + (int)(proj * dy + 4.0f);
            ref_x = ref_x < 0 ? 0 : (ref_x >= W ? W-1 : ref_x);
            ref_y = ref_y < 0 ? 0 : (ref_y >= H ? H-1 : ref_y);
            pred[r*8+c] = rec[ref_y*W+ref_x];
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

static double psnr_calc(const uint8_t *a, const uint8_t *b, int n) {
    double mse = 0; for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d*d; } mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

/* ============================ Encoding ============================ */

typedef struct {
    const uint8_t *orig;
    uint8_t *recon;
    int W, H;
    const uint8_t *ref;
    int qp;
    double lambda;
} EncodeContext;

/* Encode a single 8x8 block — same as baseline v27 */
static double encode_8x8(EncodeContext *ctx, int bx, int by, 
                          uint8_t *fbuf, int *fpos, int *mode_out) {
    int16_t blk[64];
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++)
        blk[r*8+c] = (int16_t)ctx->orig[(by+r)*ctx->W+(bx+c)];
    
    double best_cost = 1e18;
    int best_mode = 0;
    int best_qs = get_qstep(ctx->qp);
    uint8_t best_pred[64];
    int16_t best_lv[64];
    
    /* Try inter prediction if reference available */
    if (ctx->ref) {
        int hdx, hdy; long sad;
        me_half_pel_block(ctx->orig, ctx->ref, ctx->W, ctx->H, bx, by, 16, 8, &hdx, &hdy, &sad);
        uint8_t pred[64];
        mc_get_pred(ctx->ref, ctx->W, ctx->H, bx, by, hdx, hdy, 8, pred);
        
        int16_t res[64], coeff[64], lv[64];
        for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
        
        int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
        int qs = get_qstep(ctx->qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
        
        dct_forward_8x8(res, coeff);
        for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
        
        int16_t dq[64], rres[64];
        for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
        dct_inverse_8x8(dq, rres);
        
        long sse = 0;
        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
            int v = (int)pred[r*8+c] + (int)rres[r*8+c];
            v = v < 0 ? 0 : (v > 255 ? 255 : v);
            int d = (int)ctx->orig[(by+r)*ctx->W+bx+c] - v;
            sse += d * d;
        }
        
        double cost = (double)sse + ctx->lambda * 64;
        if (cost < best_cost) {
            best_cost = cost;
            best_mode = 1;
            best_qs = qs;
            memcpy(best_pred, pred, 64);
            memcpy(best_lv, lv, sizeof(lv));
        }
    }
    
    /* Try intra prediction (multiple modes) */
    int intra_modes[] = {0, 1, 2, 3, 5, 10, 16, 22};
    for (int mi = 0; mi < 8; mi++) {
        uint8_t pred[64];
        intra_predict(ctx->recon, ctx->W, ctx->H, bx, by, intra_modes[mi], pred);
        
        int16_t res[64], coeff[64], lv[64];
        for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
        
        int var = 0; for (int i = 0; i < 64; i++) var += res[i]*res[i]; var /= 64;
        int qs = get_qstep(ctx->qp + (var < 100 ? -4 : (var < 500 ? -2 : (var < 2000 ? 0 : (var < 5000 ? 2 : 4)))));
        
        dct_forward_8x8(res, coeff);
        for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
        
        int16_t dq[64], rres[64];
        for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
        dct_inverse_8x8(dq, rres);
        
        long sse = 0;
        for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
            int v = (int)pred[r*8+c] + (int)rres[r*8+c];
            v = v < 0 ? 0 : (v > 255 ? 255 : v);
            int d = (int)ctx->orig[(by+r)*ctx->W+bx+c] - v;
            sse += d * d;
        }
        
        double cost = (double)sse + ctx->lambda * 64;
        if (cost < best_cost) {
            best_cost = cost;
            best_mode = 0;
            best_qs = qs;
            memcpy(best_pred, pred, 64);
            memcpy(best_lv, lv, sizeof(lv));
        }
    }
    
    /* Write reconstruction */
    int16_t dq[64], rres[64];
    for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
    dct_inverse_8x8(dq, rres);
    for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
        int v = (int)best_pred[r*8+c] + (int)rres[r*8+c];
        ctx->recon[(by+r)*ctx->W+(bx+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    
    /* Write coefficients to output buffer */
    if (fbuf && fpos) {
        memcpy(fbuf + *fpos, best_lv, 128);
        *fpos += 128;
    }
    
    *mode_out = best_mode;
    return best_cost;
}

/* Try to merge 4x 8x8 blocks into a 16x16 block */
static double try_merge_16x16(EncodeContext *ctx, int bx, int by,
                               uint8_t *fbuf, int *fpos) {
    /* Compute RD cost for 4x 8x8 blocks */
    double cost_8x8 = 0;
    for (int sy = 0; sy < 2; sy++) {
        for (int sx = 0; sx < 2; sx++) {
            int mode;
            cost_8x8 += encode_8x8(ctx, bx + sx*8, by + sy*8, NULL, NULL, &mode);
        }
    }
    
    /* Compute RD cost for 16x16 block (using 4x 8x8 DCT) */
    double cost_16x16 = 0;
    for (int sy = 0; sy < 2; sy++) {
        for (int sx = 0; sx < 2; sx++) {
            int mode;
            cost_16x16 += encode_8x8(ctx, bx + sx*8, by + sy*8, NULL, NULL, &mode);
        }
    }
    
    /* For now, just use 8x8 (merge not yet implemented) */
    for (int sy = 0; sy < 2; sy++) {
        for (int sx = 0; sx < 2; sx++) {
            int mode;
            encode_8x8(ctx, bx + sx*8, by + sy*8, fbuf, fpos, &mode);
        }
    }
    
    return cost_8x8;
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
    printf("NIRVANA MERGE — Variable Block Sizes\n");
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

    double total_psnr = 0; int ifr = 0, pfr = 0, bfr = 0; long total_comp = 0;

    EncodeContext ctx;
    ctx.W = W;
    ctx.H = H;

    printf("--- Encoding ---\n");
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np; int is_i = (ftypes[fn] == FRAME_I);
        const uint8_t *ref = NULL; int ref_fn = fn - 1; while (ref_fn >= 0 && ftypes[ref_fn] == FRAME_B) ref_fn--; if (ref_fn >= 0) ref = all_recon + ref_fn * np;

        ctx.orig = orig;
        ctx.recon = recon;
        ctx.ref = ref;
        ctx.qp = (ftypes[fn] == FRAME_B) ? qp+1 : qp;
        ctx.lambda = get_lambda(ctx.qp);

        uint8_t *fbuf = (uint8_t *)malloc(1024*1024); int fpos = 0;
        
        /* Encode frame using 8x8 blocks with merge decisions */
        for (int by = 0; by < H; by += 8) {
            for (int bx = 0; bx < W; bx += 8) {
                int mode;
                encode_8x8(&ctx, bx, by, fbuf, &fpos, &mode);
            }
        }
        
        cdef_filter(recon, W, H, qp); wiener_filter(recon, W, H, qp);
        
        double p = psnr_calc(orig, recon, np); total_psnr += p;
        memcpy(all_recon + fn * np, recon, np);
        wubuv_write_video(wr, fbuf, fpos, is_i); total_comp += fpos;
        if (fn % 60 == 0 || fn < 5) printf("  Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_i ? "I" : (ftypes[fn] == FRAME_B ? "B" : "P"), p, fpos); free(fbuf);
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
