/* wubuv_decode.c — WUBV Decoder with actual decompression */
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
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    dct_basis[u][v][x][y] = cu * cv *
                        cosf((2*x+1)*u*3.14159265f/16.0f) *
                        cosf((2*y+1)*v*3.14159265f/16.0f);
        }
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

/* ===== Quantization ===== */
static inline int dequantize(int level, int qstep) {
    return level * qstep;
}

/* ===== Inverse Exp-Golomb ===== */
static int eg_decode(FILE* f, int* bits_read) {
    int k = 0;
    int bit;
    
    /* Count leading zeros */
    while ((bit = fgetc(f)) != EOF) {
        if (bit == 1) break;
        k++;
    }
    
    if (bit == EOF) return 0;
    
    /* Read k more bits for the value */
    int value = (1 << k);
    for (int i = 0; i < k; i++) {
        bit = fgetc(f);
        value |= (bit << (k - 1 - i));
    }
    
    *bits_read = 2 * k + 1;
    
    /* Convert to signed */
    if (value & 1) return (value + 1) / 2;
    else return -value / 2;
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

/* ===== Decode one frame ===== */
static int decode_frame(FILE* f, uint8_t* recon, int W, int H, int qstep) {
    uint8_t type;
    uint32_t size;
    
    type = fgetc(f);
    if (type == EOF) return -1;
    fread(&size, 4, 1, f);
    
    if (type == 0 || type == 1) { /* KEY or INTER */
        uint16_t frame_w, frame_h;
        fread(&frame_w, 2, 1, f);
        fread(&frame_h, 2, 1, f);
        
        int nbx = frame_w / 8, nby = frame_h / 8;
        
        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx * 8, y0 = by * 8;
                
                /* Get prediction */
                uint8_t pred[64];
                intra_dc_predict(recon, frame_w, frame_h, x0, y0, pred);
                
                /* Read and dequantize coefficients */
                int16_t dq[64], recon_res[64];
                int bits_read = 0;
                
                for (int i = 0; i < 64; i++) {
                    int level = eg_decode(f, &bits_read);
                    dq[i] = (int16_t)dequantize(level, qstep);
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
        uint16_t num_samples;
        fread(&num_samples, 2, 1, f);
        /* Skip audio for now */
        fseek(f, num_samples * sizeof(float), SEEK_CUR);
    }
    
    return 0;
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
    
    /* Get header */
    FILE* f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    
    WubuvHeader hdr;
    fread(&hdr, sizeof(WubuvHeader), 1, f);
    
    printf("WUBV Decoder\n");
    printf("  Input: %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Resolution: %dx%d @ %d fps\n", hdr.width, hdr.height, hdr.fps);
    printf("  Frames: %d\n", hdr.frame_count);
    printf("  QP: %d\n\n", qp);
    
    int qstep = 40; /* Default for QP 26 */
    int np = hdr.width * hdr.height;
    
    uint8_t* recon = malloc(np);
    memset(recon, 0, np);
    
    FILE* fout = fopen(output, "wb");
    
    int frame = 0;
    while (decode_frame(f, recon, hdr.width, hdr.height, qstep) == 0) {
        fwrite(recon, 1, np, fout);
        frame++;
        
        if (frame % 50 == 0)
            printf("  Decoded %d frames\n", frame);
    }
    
    printf("\nDecoded %d frames\n", frame);
    
    fclose(f);
    fclose(fout);
    free(recon);
    
    return 0;
}
