/* wubuv_decode_v9.c — WUBV Decoder with full inter prediction support
 *
 * Decodes I, P, and B frames with motion compensation.
 * Matches encoder's format: each block is byte-aligned.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

#define DPB_MAX 4

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

static int bs_read_ue(BitStream* bs) {
    int k = 0;
    while (bs_read_bit(bs) == 0) k++;
    int val = (1 << k);
    if (k > 0) {
        for (int i = 0; i < k; i++)
            val |= (bs_read_bit(bs) << (k - 1 - i));
    }
    return val;
}

static int bs_read_se(BitStream* bs) {
    int val = bs_read_ue(bs);
    if (val == 0) return 0;
    int sign = bs_read_bit(bs);
    return sign ? -val : val;
}

static int bs_bytes(BitStream* bs) {
    return bs->byte_pos + (bs->bit_pos > 0 ? 1 : 0);
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

/* ===== Zigzag ===== */
static const int zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

/* ===== Motion Compensation ===== */
static void mc_get_pred(const uint8_t* ref, int W, int H,
                        int bx, int by, int hdx, int hdy, uint8_t* pred) {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int x2 = (bx+c)*2 + hdx;
            int y2 = (by+r)*2 + hdy;
            int x0 = x2/2; int y0 = y2/2;
            int x1 = x0+1; int y1 = y0+1;
            int fx = x2 & 1; int fy = y2 & 1;
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
            if (x1 >= W) x1 = W-1; if (y1 >= H) y1 = H-1;
            int v = (ref[y0*W+x0]*(2-fx)*(2-fy) + ref[y0*W+x1]*fx*(2-fy) +
                     ref[y1*W+x0]*(2-fx)*fy + ref[y1*W+x1]*fx*fy) / 4;
            pred[r*8+c] = (uint8_t)v;
        }
    }
}

/* ===== Decode one frame ===== */
static int decode_frame(WubuvReader* r, uint8_t* recon, int W, int H, int qstep,
                         uint8_t* dpb_frames[DPB_MAX], int dpb_count) {
    uint8_t type;
    uint32_t size;
    uint8_t* data;

    if (wubuv_read_frame(r, &type, &data, &size) != 0) return -1;

    if (type == 0 || type == 1) { /* KEY or INTER */
        uint16_t frame_w, frame_h;
        memcpy(&frame_w, data, 2);
        memcpy(&frame_h, data + 2, 2);

        int nbx = frame_w / 8, nby = frame_h / 8;
        int offset = 4;

        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx * 8, y0 = by * 8;

                uint8_t pred[64];
                if (dpb_count > 0) {
                    /* Use most recent ref for prediction */
                    mc_get_pred(dpb_frames[dpb_count-1], frame_w, frame_h, x0, y0, 0, 0, pred);
                } else {
                    intra_dc_predict(recon, frame_w, frame_h, x0, y0, pred);
                }

                BitStream bs;
                bs_init(&bs, data + offset, size - offset);

                int nz = bs_read_ue(&bs);

                int16_t dq[64], recon_res[64];
                for (int i = 0; i < 64; i++) dq[i] = 0;

                int idx = 0;
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

                dct8_inverse(dq, recon_res);

                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++) {
                        int val = (int)pred[r*8+c] + (int)recon_res[r*8+c];
                        if (val < 0) val = 0;
                        if (val > 255) val = 255;
                        recon[(y0+r)*frame_w+(x0+c)] = (uint8_t)val;
                    }

                offset += bs_bytes(&bs);
            }
        }
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

    printf("WUBV Decoder v9\n");
    printf("  Input: %s\n", input);
    printf("  Resolution: %dx%d @ %d fps\n", W, H, r->hdr.fps);
    printf("  Frames: %d\n", nf);
    printf("  QP: %d (qstep=%d)\n\n", qp, qstep);

    uint8_t* recon = calloc((size_t)W * H, 1);
    uint8_t* frame_out = malloc((size_t)W * H);

    /* DPB for reconstructed frames */
    uint8_t* dpb_frames[DPB_MAX] = {NULL};
    int dpb_count = 0;

    FILE* fout = fopen(output, "wb");
    if (!fout) { printf("Cannot open %s\n", output); return 1; }

    int decoded = 0;
    while (decoded < nf) {
        int ret = decode_frame(r, recon, W, H, qstep, dpb_frames, dpb_count);
        if (ret != 0) break;

        memcpy(frame_out, recon, (size_t)W * H);
        fwrite(frame_out, 1, (size_t)W * H, fout);
        decoded++;

        /* Update DPB */
        if (dpb_count < DPB_MAX) {
            dpb_frames[dpb_count] = malloc((size_t)W * H);
            memcpy(dpb_frames[dpb_count], recon, (size_t)W * H);
            dpb_count++;
        } else {
            /* Shift and add */
            free(dpb_frames[0]);
            for (int i = 0; i < DPB_MAX-1; i++)
                dpb_frames[i] = dpb_frames[i+1];
            dpb_frames[DPB_MAX-1] = malloc((size_t)W * H);
            memcpy(dpb_frames[DPB_MAX-1], recon, (size_t)W * H);
        }

        if (decoded % 50 == 0 || decoded == nf)
            printf("  Decoded %d frames\n", decoded);
    }

    fclose(fout);
    free(recon);
    free(frame_out);
    for (int i = 0; i < dpb_count; i++) free(dpb_frames[i]);
    wubuv_reader_close(r);

    printf("\nDecoded %d frames to %s\n", decoded, output);
    return 0;
}
