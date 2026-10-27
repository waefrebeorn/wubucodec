/* wubuv_encode_v8_recon.c — Encoder variant that always outputs reconstruction */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

/* Copy all the functions from wubuv_encode_v8.c */
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

static const int zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

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

static int get_qstep(int qp) {
    static const int t[52]={
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722
    };
    if(qp<0)qp=0; if(qp>51)qp=51;
    return t[qp];
}

/* Main: encode with recon output */
int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s input.raw output.wubv recon.raw [options]\n", argv[0]);
        return 1;
    }

    const char *input = argv[1], *output = argv[2], *recon_path = argv[3];
    int W = 320, H = 240, fps = 24, qp = 26, gop_size = 30, b_frames = 2;

    /* Parse remaining args */
    for (int i = 4; i < argc; i += 2) {
        if (i+1 >= argc) break;
        if (strcmp(argv[i], "-w") == 0) W = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-h") == 0) H = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-f") == 0) fps = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-q") == 0) qp = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-g") == 0) gop_size = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-b") == 0) b_frames = atoi(argv[i+1]);
    }

    int np = W * H;
    printf("WUBV Encoder v8 (with recon output)\n");
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
    uint8_t *all_recon = (uint8_t *)calloc((size_t)nf * np, 1);

    double total_psnr = 0;
    int ifr = 0, pfr = 0, bfr = 0;
    long total_comp = 0;

    /* Simple I-frame only encoding for verification */
    printf("--- Encoding ---\n");
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np;
        int nbx = W/8, nby = H/8;
        uint8_t *fbuf = (uint8_t *)malloc(nbx*nby*256);
        int fpos = 0;

        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx*8, y0 = by*8;
                int16_t blk[64];
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];

                /* DC prediction */
                uint8_t pred[64];
                int sum = 0, cnt = 0;
                if (by > 0) for (int c = 0; c < 8; c++) { int x = bx+c; if (x < W) { sum += recon[(by-1)*W+x]; cnt++; } }
                if (bx > 0) for (int r = 0; r < 8; r++) { int y = by+r; if (y < H) { sum += recon[y*W+bx-1]; cnt++; } }
                uint8_t dc = (cnt > 0) ? (uint8_t)(sum/cnt) : 128;
                for (int i = 0; i < 64; i++) pred[i] = dc;

                int16_t res[64], coeff[64], lv[64];
                for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)pred[i]);
                dct8_forward(res, coeff);
                int qs = get_qstep(qp);
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                uint8_t bb[256]; int sz = encode_block(lv, bb, 256);
                memcpy(fbuf + fpos, bb, sz); fpos += sz;

                int16_t dq[64], rres[64];
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(lv[i], qs);
                dct8_inverse(dq, rres);
                for (int r = 0; r < 8; r++) for (int c = 0; c < 8; c++) {
                    int v = (int)pred[r*8+c] + (int)rres[r*8+c];
                    recon[(y0+r)*W+(x0+c)] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        double p = 0;
        for (int i = 0; i < np; i++) { double d = (double)orig[i] - recon[i]; p += d*d; }
        p = p / np;
        p = p > 0 ? 10.0 * log10(255.0*255.0/p) : 99.0;
        total_psnr += p;
        memcpy(all_recon + fn * np, recon, np);
        wubuv_write_video(wr, fbuf, fpos, 1);
        total_comp += fpos;
        free(fbuf);
        if (fn % 30 == 0 || fn < 5) printf("  Frame %d (I): PSNR=%.1f dB, bytes=%d\n", fn, p, fpos);
    }

    wubuv_writer_close(wr);
    double avg = total_psnr / nf;
    printf("\n=== Results ===\n");
    printf("Average PSNR: %.1f dB\n", avg);
    printf("Total: %.2f MB\n", total_comp / 1048576.0);

    /* Save reconstruction */
    FILE* rf = fopen(recon_path, "wb");
    if (rf) {
        fwrite(all_recon, 1, (size_t)nf * np, rf);
        fclose(rf);
        printf("Reconstruction saved to %s\n", recon_path);
    }

    free(frames); free(recon); free(all_recon);
    return 0;
}
