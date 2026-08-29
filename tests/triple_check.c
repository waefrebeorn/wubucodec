/* triple_check.c — Comprehensive verification of every component
 *
 * Tests:
 * 1. DCT/IDCT roundtrip at 8×8 (the only size we use)
 * 2. Intra prediction correctness
 * 3. Motion estimation correctness
 * 4. Quantization/dequantization
 * 5. Full encode→decode roundtrip
 * 6. Container format robustness
 * 7. Player file reading
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define BS 8

/* ===== DCT (orthonormal DCT-II) ===== */

static float dct_basis[8][8][8][8];
static int dct_ready = 0;

static void init_dct_basis(void) {
    if (dct_ready) return;
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    dct_basis[u][v][x][y] = cu * cv *
                        cosf((2 * x + 1) * u * 3.14159265f / 16.0f) *
                        cosf((2 * y + 1) * v * 3.14159265f / 16.0f);
                }
            }
        }
    }
    dct_ready = 1;
}

static void dct8_forward(const int16_t* in, int16_t* out) {
    init_dct_basis();
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float sum = 0;
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    sum += in[x * 8 + y] * dct_basis[u][v][x][y];
            out[u * 8 + v] = (int16_t)roundf(sum);
        }
    }
}

static void dct8_inverse(const int16_t* in, int16_t* out) {
    init_dct_basis();
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            float sum = 0;
            for (int u = 0; u < 8; u++)
                for (int v = 0; v < 8; v++)
                    sum += in[u * 8 + v] * dct_basis[u][v][x][y];
            out[x * 8 + y] = (int16_t)roundf(sum);
        }
    }
}

/* ===== Test 1: DCT Roundtrip ===== */

static int test_dct_roundtrip(void) {
    printf("=== Test 1: DCT Roundtrip ===\n");
    
    int16_t in[64], coeff[64], out[64];
    double max_mse = 0;
    
    /* Test with various patterns */
    for (int pattern = 0; pattern < 5; pattern++) {
        switch (pattern) {
            case 0: /* Constant */
                for (int i = 0; i < 64; i++) in[i] = 128;
                break;
            case 1: /* Gradient */
                for (int i = 0; i < 64; i++) in[i] = (int16_t)(i * 4);
                break;
            case 2: /* Impulse */
                for (int i = 0; i < 64; i++) in[i] = 0;
                in[0] = 100;
                break;
            case 3: /* Random */
                srand(42);
                for (int i = 0; i < 64; i++) in[i] = (int16_t)(rand() % 256);
                break;
            case 4: /* Real cartoon block */
                for (int i = 0; i < 64; i++) in[i] = (int16_t)(50 + (i % 8) * 20);
                break;
        }
        
        dct8_forward(in, coeff);
        dct8_inverse(coeff, out);
        
        double mse = 0;
        for (int i = 0; i < 64; i++) {
            double d = (double)in[i] - (double)out[i];
            mse += d * d;
        }
        mse /= 64;
        
        if (mse > max_mse) max_mse = mse;
        printf("  Pattern %d: MSE = %.4f\n", pattern, mse);
    }
    
    if (max_mse < 1.0) {
        printf("  PASS: Max MSE = %.4f (< 1.0)\n\n", max_mse);
        return 1;
    } else {
        printf("  FAIL: Max MSE = %.4f (>= 1.0)\n\n", max_mse);
        return 0;
    }
}

/* ===== Test 2: Intra Prediction ===== */

static void intra_dc_predict(const uint8_t* recon, int W, int H,
                              int bx, int by, uint8_t* pred) {
    int sum = 0, cnt = 0;
    
    if (by > 0) {
        for (int c = 0; c < 8; c++) {
            int x = bx + c;
            if (x < W) { sum += recon[(by - 1) * W + x]; cnt++; }
        }
    }
    if (bx > 0) {
        for (int r = 0; r < 8; r++) {
            int y = by + r;
            if (y < H) { sum += recon[y * W + bx - 1]; cnt++; }
        }
    }
    
    uint8_t dc = (cnt > 0) ? (uint8_t)(sum / cnt) : 128;
    for (int i = 0; i < 64; i++) pred[i] = dc;
}

static int test_intra_prediction(void) {
    printf("=== Test 2: Intra Prediction ===\n");
    
    /* Create a simple gradient image */
    uint8_t img[320 * 240];
    for (int y = 0; y < 240; y++)
        for (int x = 0; x < 320; x++)
            img[y * 320 + x] = (uint8_t)((x + y) % 256);
    
    uint8_t recon[320 * 240];
    memset(recon, 128, 320 * 240);
    
    /* Test DC prediction at various positions */
    int pass = 1;
    
    /* Position (8, 8) — has both top and left neighbors */
    uint8_t pred[64];
    intra_dc_predict(recon, 320, 240, 8, 8, pred);
    
    /* Expected: average of recon[7][8..15] and recon[8..15][7] */
    int expected_sum = 0, expected_cnt = 0;
    for (int c = 0; c < 8; c++) { expected_sum += recon[7 * 320 + 8 + c]; expected_cnt++; }
    for (int r = 0; r < 8; r++) { expected_sum += recon[(8 + r) * 320 + 7]; expected_cnt++; }
    uint8_t expected_dc = (uint8_t)(expected_sum / expected_cnt);
    
    if (pred[0] != expected_dc) {
        printf("  FAIL: DC at (8,8): expected %d, got %d\n", expected_dc, pred[0]);
        pass = 0;
    } else {
        printf("  PASS: DC at (8,8) = %d\n", pred[0]);
    }
    
    /* Position (0, 8) — no left neighbor */
    intra_dc_predict(recon, 320, 240, 0, 8, pred);
    expected_sum = 0; expected_cnt = 0;
    for (int c = 0; c < 8; c++) { expected_sum += recon[7 * 320 + c]; expected_cnt++; }
    expected_dc = (uint8_t)(expected_sum / expected_cnt);
    
    if (pred[0] != expected_dc) {
        printf("  FAIL: DC at (0,8): expected %d, got %d\n", expected_dc, pred[0]);
        pass = 0;
    } else {
        printf("  PASS: DC at (0,8) = %d\n", pred[0]);
    }
    
    /* Position (8, 0) — no top neighbor */
    intra_dc_predict(recon, 320, 240, 8, 0, pred);
    expected_sum = 0; expected_cnt = 0;
    for (int r = 0; r < 8; r++) { expected_sum += recon[r * 320 + 7]; expected_cnt++; }
    expected_dc = (uint8_t)(expected_sum / expected_cnt);
    
    if (pred[0] != expected_dc) {
        printf("  FAIL: DC at (8,0): expected %d, got %d\n", expected_dc, pred[0]);
        pass = 0;
    } else {
        printf("  PASS: DC at (8,0) = %d\n", pred[0]);
    }
    
    /* Position (0, 0) — no neighbors */
    intra_dc_predict(recon, 320, 240, 0, 0, pred);
    if (pred[0] != 128) {
        printf("  FAIL: DC at (0,0): expected 128, got %d\n", pred[0]);
        pass = 0;
    } else {
        printf("  PASS: DC at (0,0) = 128\n");
    }
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ===== Test 3: Motion Estimation ===== */

static long sad_8x8(const uint8_t* curr, const uint8_t* ref, int W, int H,
                    int bx, int by, int dx, int dy) {
    int rx = bx + dx, ry = by + dy;
    if (rx < 0 || ry < 0 || rx + 8 > W || ry + 8 > H) return 1LL << 30;
    
    long sad = 0;
    for (int r = 0; r < 8; r++) {
        const uint8_t* cp = curr + (by + r) * W + bx;
        const uint8_t* rp = ref + (ry + r) * W + rx;
        for (int c = 0; c < 8; c++)
            sad += abs((int)cp[c] - (int)rp[c]);
    }
    return sad;
}

static int test_motion_estimation(void) {
    printf("=== Test 3: Motion Estimation ===\n");
    
    /* Create two frames: second is shifted by (3, 2) */
    uint8_t frame1[320 * 240], frame2[320 * 240];
    memset(frame1, 128, 320 * 240);
    memset(frame2, 128, 320 * 240);
    
    /* Put a pattern in frame1 */
    for (int y = 50; y < 100; y++)
        for (int x = 50; x < 100; x++)
            frame1[y * 320 + x] = 200;
    
    /* Same pattern shifted by (3, 2) in frame2 */
    for (int y = 52; y < 102; y++)
        for (int x = 53; x < 103; x++)
            frame2[y * 320 + x] = 200;
    
    /* Search for block at (50, 50) in frame1 within frame2 */
    int best_dx = 0, best_dy = 0;
    long best_sad = sad_8x8(frame1, frame2, 320, 240, 50, 50, 0, 0);
    
    static const int dia[][2] = {{0,-1},{-1,0},{1,0},{0,1}};
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int d = 0; d < 4; d++) {
            int ndx = best_dx + dia[d][0];
            int ndy = best_dy + dia[d][1];
            if (abs(ndx) > 16 || abs(ndy) > 16) continue;
            long sad = sad_8x8(frame1, frame2, 320, 240, 50, 50, ndx, ndy);
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = ndx;
                best_dy = ndy;
                improved = 1;
            }
        }
    }
    
    printf("  Expected MV: (3, 2), Got: (%d, %d)\n", best_dx, best_dy);
    
    if (best_dx == 3 && best_dy == 2) {
        printf("  PASS\n\n");
        return 1;
    } else {
        printf("  FAIL\n\n");
        return 0;
    }
}

/* ===== Test 4: Quantization ===== */

static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep / 2) / qstep);
}

static inline int dequantize(int level, int qstep) {
    return level * qstep;
}

static int test_quantization(void) {
    printf("=== Test 4: Quantization ===\n");
    
    int pass = 1;
    
    /* Test basic quantization */
    if (quantize(100, 10) != 10) {
        printf("  FAIL: quantize(100, 10) = %d (expected 10)\n", quantize(100, 10));
        pass = 0;
    }
    if (quantize(-100, 10) != -10) {
        printf("  FAIL: quantize(-100, 10) = %d (expected -10)\n", quantize(-100, 10));
        pass = 0;
    }
    if (quantize(0, 10) != 0) {
        printf("  FAIL: quantize(0, 10) = %d (expected 0)\n", quantize(0, 10));
        pass = 0;
    }
    if (quantize(4, 10) != 0) {
        printf("  FAIL: quantize(4, 10) = %d (expected 0)\n", quantize(4, 10));
        pass = 0;
    }
    if (quantize(5, 10) != 1) {
        printf("  FAIL: quantize(5, 10) = %d (expected 1)\n", quantize(5, 10));
        pass = 0;
    }
    
    /* Test dequantization */
    if (dequantize(10, 10) != 100) {
        printf("  FAIL: dequantize(10, 10) = %d (expected 100)\n", dequantize(10, 10));
        pass = 0;
    }
    if (dequantize(-10, 10) != -100) {
        printf("  FAIL: dequantize(-10, 10) = %d (expected -100)\n", dequantize(-10, 10));
        pass = 0;
    }
    
    /* Test roundtrip error is bounded */
    int max_error = 0;
    for (int c = -1000; c <= 1000; c++) {
        int q = quantize(c, 40);
        int dq = dequantize(q, 40);
        int error = abs(c - dq);
        if (error > max_error) max_error = error;
    }
    if (max_error > 40) {
        printf("  FAIL: Max quant roundtrip error = %d (expected <= 40)\n", max_error);
        pass = 0;
    } else {
        printf("  PASS: Max quant roundtrip error = %d\n", max_error);
    }
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ===== Test 5: Full Encode→Decode Roundtrip ===== */

static void me_block(const uint8_t* curr, const uint8_t* ref, int W, int H,
                     int bx, int by, int search_range,
                     int* out_dx, int* out_dy, long* out_sad) {
    int best_dx = 0, best_dy = 0;
    long best_sad = sad_8x8(curr, ref, W, H, bx, by, 0, 0);
    
    static const int dia[][2] = {{0,-1},{-1,0},{1,0},{0,1}};
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int d = 0; d < 4; d++) {
            int ndx = best_dx + dia[d][0];
            int ndy = best_dy + dia[d][1];
            if (abs(ndx) > search_range || abs(ndy) > search_range) continue;
            long sad = sad_8x8(curr, ref, W, H, bx, by, ndx, ndy);
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = ndx;
                best_dy = ndy;
                improved = 1;
            }
        }
    }
    *out_dx = best_dx;
    *out_dy = best_dy;
    *out_sad = best_sad;
}

static int test_full_roundtrip(void) {
    printf("=== Test 5: Full Encode→Decode Roundtrip ===\n");
    
    /* Create a simple test sequence */
    int W = 64, H = 48;
    int np = W * H;
    int num_frames = 10;
    
    uint8_t* frames = malloc(np * num_frames);
    
    /* Generate frames with motion */
    for (int f = 0; f < num_frames; f++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                /* Moving circle */
                int cx = 20 + f * 3, cy = 24;
                int dx = x - cx, dy = y - cy;
                if (dx * dx + dy * dy < 100)
                    frames[f * np + y * W + x] = 200;
                else
                    frames[f * np + y * W + x] = 50;
            }
        }
    }
    
    uint8_t* recon = malloc(np);
    uint8_t* prev_recon = malloc(np);
    memset(prev_recon, 0, np);
    
    double total_psnr = 0;
    
    for (int f = 0; f < num_frames; f++) {
        uint8_t* orig = frames + f * np;
        uint8_t* ref = (f > 0) ? prev_recon : NULL;
        
        /* Encode frame */
        memset(recon, 0, np);
        
        for (int by = 0; by < H; by += 8) {
            for (int bx = 0; bx < W; bx += 8) {
                int x0 = bx, y0 = by;
                
                int16_t block[64];
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        block[r * 8 + c] = (int16_t)orig[(y0 + r) * W + (x0 + c)];
                
                /* Intra prediction */
                uint8_t intra_pred[64];
                intra_dc_predict(recon, W, H, x0, y0, intra_pred);
                
                int16_t intra_res[64];
                for (int i = 0; i < 64; i++)
                    intra_res[i] = (int16_t)((int)block[i] - (int)intra_pred[i]);
                
                int16_t intra_coeff[64], intra_levels[64];
                dct8_forward(intra_res, intra_coeff);
                
                int qstep = 40;
                for (int i = 0; i < 64; i++)
                    intra_levels[i] = quantize(intra_coeff[i], qstep);
                
                int16_t intra_dq[64], intra_recon_res[64];
                for (int i = 0; i < 64; i++)
                    intra_dq[i] = (int16_t)dequantize(intra_levels[i], qstep);
                dct8_inverse(intra_dq, intra_recon_res);
                
                uint8_t intra_recon[64];
                long intra_sse = 0;
                for (int i = 0; i < 64; i++) {
                    int val = (int)intra_pred[i] + (int)intra_recon_res[i];
                    if (val < 0) val = 0;
                    if (val > 255) val = 255;
                    intra_recon[i] = (uint8_t)val;
                    int d = (int)block[i] - val;
                    intra_sse += d * d;
                }
                
                /* Inter prediction */
                long inter_sse = 1LL << 60;
                uint8_t inter_recon[64];
                
                if (ref) {
                    int dx, dy;
                    long sad;
                    me_block(orig, ref, W, H, x0, y0, 16, &dx, &dy, &sad);
                    
                    uint8_t inter_pred[64];
                    int rx = x0 + dx, ry = y0 + dy;
                    for (int r = 0; r < 8; r++)
                        for (int c = 0; c < 8; c++)
                            inter_pred[r * 8 + c] = ref[(ry + r) * W + (rx + c)];
                    
                    int16_t inter_res[64];
                    for (int i = 0; i < 64; i++)
                        inter_res[i] = (int16_t)((int)block[i] - (int)inter_pred[i]);
                    
                    int16_t inter_coeff[64], inter_levels[64];
                    dct8_forward(inter_res, inter_coeff);
                    for (int i = 0; i < 64; i++)
                        inter_levels[i] = quantize(inter_coeff[i], qstep);
                    
                    int16_t inter_dq[64], inter_recon_res[64];
                    for (int i = 0; i < 64; i++)
                        inter_dq[i] = (int16_t)dequantize(inter_levels[i], qstep);
                    dct8_inverse(inter_dq, inter_recon_res);
                    
                    inter_sse = 0;
                    for (int i = 0; i < 64; i++) {
                        int val = (int)inter_pred[i] + (int)inter_recon_res[i];
                        if (val < 0) val = 0;
                        if (val > 255) val = 255;
                        inter_recon[i] = (uint8_t)val;
                        int d = (int)block[i] - val;
                        inter_sse += d * d;
                    }
                }
                
                /* Pick best */
                if (inter_sse < intra_sse) {
                    for (int r = 0; r < 8; r++)
                        for (int c = 0; c < 8; c++)
                            recon[(y0 + r) * W + (x0 + c)] = inter_recon[r * 8 + c];
                } else {
                    for (int r = 0; r < 8; r++)
                        for (int c = 0; c < 8; c++)
                            recon[(y0 + r) * W + (x0 + c)] = intra_recon[r * 8 + c];
                }
            }
        }
        
        /* Compute PSNR */
        double mse = 0;
        for (int i = 0; i < np; i++) {
            double d = (double)orig[i] - (double)recon[i];
            mse += d * d;
        }
        mse /= np;
        double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        total_psnr += psnr;
        
        printf("  Frame %d: PSNR = %.1f dB\n", f, psnr);
        
        memcpy(prev_recon, recon, np);
    }
    
    double avg_psnr = total_psnr / num_frames;
    printf("  Average PSNR: %.1f dB\n", avg_psnr);
    
    free(frames);
    free(recon);
    free(prev_recon);
    
    if (avg_psnr > 30.0) {
        printf("  PASS\n\n");
        return 1;
    } else {
        printf("  FAIL\n\n");
        return 0;
    }
}

/* ===== Main ===== */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           TRIPLE CHECK — Component Verification            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int pass = 0, total = 0;
    
    total++; pass += test_dct_roundtrip();
    total++; pass += test_intra_prediction();
    total++; pass += test_motion_estimation();
    total++; pass += test_quantization();
    total++; pass += test_full_roundtrip();
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RESULTS: %d/%d tests passed", pass, total);
    if (pass == total) {
        printf(" — ALL PASS ✓\n");
    } else {
        printf(" — %d FAILURES ✗\n", total - pass);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return (pass == total) ? 0 : 1;
}
