/* clean_enc.c: Clean, working encoder test
 *
 * Tests the raw encoder step by step to find what's broken.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_encoder.h"
#include "wubu_transform.h"
#include "wubu_intra.h"

#define W 320
#define H 240
#define BS 8

static double frame_psnr(const uint8_t* a, const uint8_t* b, int n) {
    double mse = 0;
    for(int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
}

/* Test 1: Just DCT + IDCT (no quantization) */
static void test_dct_only() {
    printf("=== Test 1: DCT + IDCT (no quant) ===\n");
    uint8_t orig[64];
    int16_t res[64], coeffs[64], dq[64], recon[64];
    
    // Create a simple gradient block
    for(int i = 0; i < 64; i++) orig[i] = (uint8_t)(i * 4);
    
    // residual = orig - 128 (DC removal)
    for(int i = 0; i < 64; i++) res[i] = (int16_t)((int)orig[i] - 128);
    
    // Forward DCT
    wubu_tr_forward(res, coeffs, BS);
    
    // No quantization - direct inverse
    wubu_tr_inverse(coeffs, recon, BS);
    
    // Add DC back
    printf("Orig vs Recon (DC removed):\n");
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            int o = orig[i*8+j];
            int r = recon[i*8+j] + 128;
            printf("%3d/%3d ", o, r);
        }
        printf("\n");
    }
    
    double mse = 0;
    for(int i = 0; i < 64; i++) {
        int r = recon[i] + 128;
        if(r < 0) r = 0; if(r > 255) r = 255;
        int d = (int)orig[i] - r;
        mse += d*d;
    }
    printf("MSE: %.2f\n\n", mse / 64);
}

/* Test 2: DCT + quant + IDCT */
static void test_quant() {
    printf("=== Test 2: DCT + Quant + IDCT ===\n");
    uint8_t orig[64];
    int16_t res[64], coeffs[64], dq[64], recon[64];
    
    for(int i = 0; i < 64; i++) orig[i] = (uint8_t)(i * 4);
    
    for(int i = 0; i < 64; i++) res[i] = (int16_t)((int)orig[i] - 128);
    
    wubu_tr_forward(res, coeffs, BS);
    
    printf("Coefficients (first 8): ");
    for(int i = 0; i < 8; i++) printf("%d ", coeffs[i]);
    printf("\n");
    
    // Try different quantization steps
    for(int qstep = 1; qstep <= 20; qstep *= 2) {
        for(int i = 0; i < 64; i++) {
            // Signed quantization
            int c = coeffs[i];
            int sign = (c >= 0) ? 1 : -1;
            dq[i] = (int16_t)(sign * (abs(c) + qstep/2) / qstep * qstep);
        }
        
        wubu_tr_inverse(dq, recon, BS);
        
        double mse = 0;
        for(int i = 0; i < 64; i++) {
            int r = recon[i] + 128;
            if(r < 0) r = 0; if(r > 255) r = 255;
            int d = (int)orig[i] - r;
            mse += d*d;
        }
        printf("  qstep=%2d: MSE=%.1f\n", qstep, mse / 64);
    }
    printf("\n");
}

/* Test 3: Intra prediction quality */
static void test_intra() {
    printf("=== Test 3: Intra Prediction ===\n");
    
    uint8_t img[W*H];
    uint8_t recon[W*H];
    memset(recon, 128, W*H);
    
    // Create a simple gradient image
    for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
            img[y*W+x] = (uint8_t)((x + y) % 256);
    
    // Try intra prediction for one block
    uint8_t pred[64], org_blk[64];
    int bx = 1, by = 1;
    
    for(int r = 0; r < 8; r++)
        for(int c = 0; c < 8; c++)
            org_blk[r*8+c] = img[(by*8+r)*W + (bx*8+c)];
    
    // Vertical prediction
    uint8_t vert[64];
    memset(vert, 0, 64);
    for(int c = 0; c < 8; c++) {
        uint8_t v = img[(by-1)*8*W + bx*8+c]; // top row
        for(int r = 0; r < 8; r++) vert[r*8+c] = v;
    }
    
    // Horizontal prediction
    uint8_t horiz[64];
    memset(horiz, 0, 64);
    for(int r = 0; r < 8; r++) {
        uint8_t v = img[(by+r)*W + bx-1]; // left col
        for(int c = 0; c < 8; c++) horiz[r*8+c] = v;
    }
    
    printf("Orig block:\n");
    for(int r = 0; r < 8; r++) {
        for(int c = 0; c < 8; c++) printf("%3d ", org_blk[r*8+c]);
        printf("\n");
    }
    
    printf("Vert pred:\n");
    for(int r = 0; r < 8; r++) {
        for(int c = 0; c < 8; c++) printf("%3d ", vert[r*8+c]);
        printf("\n");
    }
    
    printf("Horiz pred:\n");
    for(int r = 0; r < 8; r++) {
        for(int c = 0; c < 8; c++) printf("%3d ", horiz[r*8+c]);
        printf("\n");
    }
    
    // Compute SAD
    long sad_vert = 0, sad_horiz = 0;
    for(int i = 0; i < 64; i++) {
        sad_vert += abs((int)org_blk[i] - (int)vert[i]);
        sad_horiz += abs((int)org_blk[i] - (int)horiz[i]);
    }
    printf("SAD vert=%ld, horiz=%ld\n\n", sad_vert, sad_horiz);
}

/* Test 4: Real cartoon frame encode */
static void test_real_frame() {
    printf("=== Test 4: Real Cartoon Frame ===\n");
    
    int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    
    FILE* f = fopen("/tmp/balloon.raw", "rb");
    if(!f) { printf("Cannot open\n"); return; }
    fread(frame, 1, np, f);
    fclose(f);
    
    // Check frame stats
    int min=256, max=-1;
    long sum = 0;
    for(int i = 0; i < np; i++) {
        if(frame[i] < min) min = frame[i];
        if(frame[i] > max) max = frame[i];
        sum += frame[i];
    }
    printf("Frame: min=%d, max=%d, avg=%ld\n", min, max, sum/np);
    
    // Encode
    long bits = 0;
    int rc = wubu_encode_frame(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits);
    printf("Encode: rc=%d, bits=%ld (%.2f bpp)\n", rc, bits, (double)bits/np);
    
    double psnr_val = frame_psnr(frame, recon, np);
    printf("PSNR: %.1f dB\n", psnr_val);
    
    // Check recon stats
    min=256; max=-1; sum=0;
    for(int i = 0; i < np; i++) {
        if(recon[i] < min) min = recon[i];
        if(recon[i] > max) max = recon[i];
        sum += recon[i];
    }
    printf("Recon: min=%d, max=%d, avg=%ld\n", min, max, sum/np);
    
    // Save for visual inspection
    FILE* fout = fopen("/tmp/recon_raw.raw", "wb");
    fwrite(recon, 1, np, fout);
    fclose(fout);
    
    free(frame);
    free(recon);
}

int main() {
    printf("=== Encoder Diagnostic ===\n\n");
    test_dct_only();
    test_quant();
    test_intra();
    test_real_frame();
    return 0;
}
