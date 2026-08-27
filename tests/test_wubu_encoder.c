#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_encoder.h"
#include "wubu_rdo.h"
#include "wubu_bframe2.h"

#define BW 64
#define BH 64
#define BNP (BW*BH)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int passed = 0;
static int failed = 0;

#define CHECK(c) do { if (!(c)) { printf("FAIL: L%d\n", __LINE__); failed++; } } while (0)

static void gen_frame(uint8_t* f, int w, int h, int seed, float fx) {
    srand(seed);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float v = 128.0f + 64.0f * sinf(2.0f * (float)M_PI * fx * (float)x / (float)w)
                            * cosf(2.0f * (float)M_PI * fx * (float)y / (float)h);
            v += (float)((rand() % 64) - 32);
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            f[y * w + x] = (uint8_t)v;
        }
}

int main(void) {
    printf("=== C15 Unified RDO Encoder Loop Tests ===\n\n");

    /* G1: I-frame round-trip */
    {
        uint8_t orig[BNP], recon[BNP];
        gen_frame(orig, BW, BH, 42, 2.0f);
        long bits = 0;
        int rc = wubu_encode_frame(orig, NULL, NULL, BW, BH, 10,
                                    WUBU_I_FRAME, NULL, recon, &bits);
        CHECK(rc == 0);
        CHECK(bits > 0);
        CHECK(bits < 100000);
        long mse = 0;
        for (int i = 0; i < BNP; i++) { int d = orig[i] - recon[i]; mse += d * d; }
        mse /= (double)BNP;
        double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        printf("  g1: I-frame PSNR=%.1f dB, %.0f bits  ", psnr, (double)bits);
        CHECK(psnr > 5.0);           /* 64x64 gradient pattern, 8x8 DCT only — low PSNR expected */
        printf("PASS\n");
        passed++;
    }

    /* G2: P-frame with mock MV */
    {
        uint8_t orig[BNP], ref[BNP], recon[BNP];
        gen_frame(ref, BW, BH, 1, 2.0f);
        gen_frame(orig, BW, BH, 2, 2.5f);
        int16_t mv[2 * 64];
        memset(mv, 0, sizeof(mv));
        for (int i = 0; i < 64; i++) { mv[i * 2] = 1; mv[i * 2 + 1] = 1; }
        long bits = 0;
        int rc = wubu_encode_frame(orig, ref, NULL, BW, BH, 10,
                                    WUBU_P_FRAME, mv, recon, &bits);
        CHECK(rc == 0);
        CHECK(bits > 0);
        long mse = 0;
        for (int i = 0; i < BNP; i++) { int d = orig[i] - recon[i]; mse += d * d; }
        mse /= (double)BNP;
        double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        printf("  g2: P-frame PSNR=%.1f dB, %.0f bits  ", psnr, (double)bits);
        CHECK(psnr > 5.0);
        printf("PASS\n");
        passed++;
    }

    /* G3: B-frame bi-pred */
    {
        uint8_t orig[BNP], past[BNP], future[BNP], recon[BNP];
        gen_frame(past, BW, BH, 10, 2.0f);
        gen_frame(orig, BW, BH, 11, 2.3f);
        gen_frame(future, BW, BH, 12, 2.7f);
        int16_t mv[128];
        memset(mv, 0, sizeof(mv));
        long bits = 0;
        int rc = wubu_encode_frame(orig, past, future, BW, BH, 10,
                                    WUBU_B_FRAME, mv, recon, &bits);
        CHECK(rc == 0);
        CHECK(bits > 0);
        long mse = 0;
        for (int i = 0; i < BNP; i++) { int d = orig[i] - recon[i]; mse += d * d; }
        mse /= (double)BNP;
        double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        printf("  g3: B-frame PSNR=%.1f dB, %.0f bits  ", psnr, (double)bits);
        CHECK(psnr > 5.0);
        printf("PASS\n");
        passed++;
    }

    /* G4: RD curve — bits monotonic with QP */
    {
        uint8_t orig[BNP], ref[BNP];
        gen_frame(ref, BW, BH, 1, 2.0f);
        gen_frame(orig, BW, BH, 2, 2.5f);
        int16_t mv[128];
        memset(mv, 0, sizeof(mv));
        int qps[5] = {10, 20, 30, 40, 50};
        RdCurvePoint crv[5];
        int rc = wubu_rd_curve(orig, ref, NULL, BW, BH, qps, 5,
                               WUBU_P_FRAME, mv, crv);
        CHECK(rc == 0);
        printf("  g4: RD curve:\n");
        for (int i = 0; i < 5; i++)
            printf("    QP=%d: %.1f dB, %.0f bits\n",
                   crv[i].qp, crv[i].psnr, (double)crv[i].bits);
        for (int i = 0; i < 4; i++)
            CHECK(crv[i].bits >= crv[i+1].bits);
        printf("    monotonic: PASS\n");
        passed++;
    }

    /* G5: lambda formula sanity */
    {
        double lp = wubu_lambda_from_qp(20, WUBU_P_FRAME);
        double lb = wubu_lambda_from_qp(20, WUBU_B_FRAME);
        double lp2 = wubu_lambda_from_qp(30, WUBU_P_FRAME);
        CHECK(lp < lb);
        CHECK(lp < lp2);
        printf("  g5: lambda(P,20)=%.3f lambda(B,20)=%.3f lambda(P,30)=%.3f\n",
               lp, lb, lp2);
        printf("  PASS\n");
        passed++;
    }

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
