/* wubuv_decode_v3.c — WUBV Decoder with proper bitstream parsing
 *
 * Fixes:
 * 1. Proper bitstream reading from frame payload (not raw byte reads)
 * 2. Correct exp-golomb decoding from bitstream
 * 3. Correct QP→qstep mapping
 * 4. Uses WUBV container API
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

/* ===== DCT ===== */
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

static void dct8_inverse(const int16_t* in, int16_t* out) {
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

static inline int dequantize(int level, int qstep) { return level * qstep; }

/* ===== Bitstream Reader ===== */
typedef struct {
    uint8_t* data;
    int size;
    int byte_pos;
    int bit_pos;
} BitStream;

static void bs_init(BitStream* bs, uint8_t* data, int size) {
    bs->data = data;
    bs->size = size;
    bs->byte_pos = 0;
    bs->bit_pos = 0;
}

static int bs_read_bit(BitStream* bs) {
    if (bs->byte_pos >= bs->size) return 0;
    int bit = (bs->data[bs->byte_pos] >> (7 - bs->bit_pos)) & 1;
    bs->bit_pos++;
    if (bs->bit_pos == 8) {
        bs->bit_pos = 0;
        bs->byte_pos++;
    }
    return bit;
}

static int bs_read_bits(BitStream* bs, int n) {
    int val = 0;
    for (int i = 0; i < n; i++)
        val = (val << 1) | bs_read_bit(bs);
    return val;
}

static int bs_read_ue(BitStream* bs) {
    int k = 0;
    while (bs_read_bit(bs) == 0) k++;
    int val = (1 << k);
    if (k > 0) {
        int suffix = bs_read_bits(bs, k);
        val |= suffix;
    }
    return val;
}

static int bs_read_se(BitStream* bs) {
    int val = bs_read_ue(bs);
    if (val & 1) return (val + 1) / 2;
    else return -val / 2;
}

/* ===== Intra Prediction ===== */
static void intra_dc_predict(const uint8_t* recon, int W, int H,
                              int bx, int by, uint8_t* pred) {
    int sum = 0, cnt = 0;
    if (by > 0)
        for (int c = 0; c < 8; c++) {
            int x = bx + c;
            if (x < W) { sum += recon[(by-1)*W+x]; cnt++; }
        }
    if (bx > 0)
        for (int r = 0; r < 8; r++) {
            int y = by + r;
            if (y < H) { sum += recon[y*W+bx-1]; cnt++; }
        }
    uint8_t dc = (cnt > 0) ? (uint8_t)(sum/cnt) : 128;
    for (int i = 0; i < 64; i++) pred[i] = dc;
}

static void intra_vert_predict(const uint8_t* recon, int W, int H,
                                int bx, int by, uint8_t* pred) {
    if (by == 0) { intra_dc_predict(recon, W, H, bx, by, pred); return; }
    for (int c = 0; c < 8; c++) {
        int x = bx + c;
        if (x >= W) break;
        uint8_t v = recon[(by-1)*W+x];
        for (int r = 0; r < 8; r++) pred[r*8+c] = v;
    }
}

static void intra_horiz_predict(const uint8_t* recon, int W, int H,
                                 int bx, int by, uint8_t* pred) {
    if (bx == 0) { intra_dc_predict(recon, W, H, bx, by, pred); return; }
    for (int r = 0; r < 8; r++) {
        int y = by + r;
        if (y >= H) break;
        uint8_t v = recon[y*W+bx-1];
        for (int c = 0; c < 8; c++) pred[r*8+c] = v;
    }
}

/* ===== Zigzag scan ===== */
static const int zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

/* ===== Decode one frame ===== */
static int decode_frame(WubuvReader* r, uint8_t* recon, int W, int H, int qstep) {
    uint8_t type;
    uint32_t size;
    uint8_t* data;

    if (wubuv_read_frame(r, &type, &data, &size) != 0) return -1;

    if (type == 0 || type == 1) { /* KEY or INTER */
        uint16_t frame_w, frame_h;
        memcpy(&frame_w, data, 2);
        memcpy(&frame_h, data + 2, 2);

        int nbx = frame_w / 8, nby = frame_h / 8;

        /* Initialize bitstream reader at offset 4 (after width/height) */
        BitStream bs;
        bs_init(&bs, data + 4, size - 4);

        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx * 8, y0 = by * 8;

                /* Get prediction */
                uint8_t pred[64];
                intra_dc_predict(recon, frame_w, frame_h, x0, y0, pred);

                /* Read and dequantize coefficients */
                int16_t dq[64], recon_res[64];

                /* Read nz count */
                int nz = bs_read_ue(&bs);

                /* Read run-level pairs */
                int idx = 0;
                for (int i = 0; i < 64; i++) dq[i] = 0;

                for (int i = 0; i < nz; i++) {
                    int run = bs_read_ue(&bs);
                    int level = bs_read_se(&bs);
                    idx += run;
                    if (idx < 64) {
                        int zz = zigzag[idx];
                        dq[zz] = (int16_t)dequantize(level, qstep);
                    }
                    idx++;
                }

                /* Inverse DCT */
                dct8_inverse(dq, recon_res);

                /* Reconstruct */
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++) {
                        int val = (int)pred[r*8+c] + (int)recon_res[r*8+c];
                        if (val < 0) val = 0;
                        if (val > 255) val = 255;
                        recon[(y0+r)*frame_w+(x0+c)] = (uint8_t)val;
                    }
            }
        }
    } else if (type == 2) { /* AUDIO */
        /* Skip audio */
    }

    free(data);
    return 0;
}

/* ===== QP to QStep ===== */
static int get_qstep(int qp) {
    static const int t[52]={
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722
    };
    if(qp<0)qp=0; if(qp>51)qp=51;
    return t[qp];
}

/* ===== Main ===== */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s input.wubv output.raw [qp]\n", argv[0]);
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int qp = (argc > 3) ? atoi(argv[3]) : 26;

    WubuvReader* r = wubuv_reader_open(input);
    if (!r) { printf("Cannot open %s\n", input); return 1; }

    int W = r->hdr.width;
    int H = r->hdr.height;
    int nf = r->hdr.frame_count;
    int qstep = get_qstep(qp);

    printf("WUBV Decoder v3\n");
    printf("  Input: %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Resolution: %dx%d @ %d fps\n", W, H, r->hdr.fps);
    printf("  Frames: %d\n", nf);
    printf("  QP: %d (qstep=%d)\n\n", qp, qstep);

    uint8_t* recon = calloc((size_t)W * H, 1);
    uint8_t* frame_out = malloc((size_t)W * H);

    FILE* fout = fopen(output, "wb");
    if (!fout) { printf("Cannot open %s\n", output); return 1; }

    int decoded = 0;
    while (decoded < nf) {
        int ret = decode_frame(r, recon, W, H, qstep);
        if (ret != 0) break;

        memcpy(frame_out, recon, (size_t)W * H);
        fwrite(frame_out, 1, (size_t)W * H, fout);
        decoded++;

        if (decoded % 50 == 0 || decoded == nf)
            printf("  Decoded %d frames\n", decoded);
    }

    fclose(fout);
    free(recon);
    free(frame_out);
    wubuv_reader_close(r);

    printf("\nDecoded %d frames to %s\n", decoded, output);
    return 0;
}
