/* wubu_deep_train.c — WuBuMath Deep Model Training
 *
 * Deeper model for better generalization to real video.
 * Architecture:
 *   Encoder: patch[48] → Linear[48→64] → LayerNorm → 8 flow MLP levels → latent[64]
 *   Decoder: latent[64] → Linear[64→128] → GELU → Linear[128→48]
 *
 * Trained on diverse synthetic data (random shapes, colors, motion).
 * The model learns the PHYSICS, not the content.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Larger model for better generalization */
#define IMG_H 64
#define IMG_W 64
#define PS 4
#define NPH 16
#define NPW 16
#define NP 256
#define PP 48
#define DIM 64      /* Increased from 32 */
#define LDIM 64     /* Increased from 32 */
#define LVL 8       /* Increased from 4 */
#define FH 64       /* Increased from 32 */

#define BATCH 16
#define EPOCHS 300
#define NSAMP 1024  /* More training data */

static unsigned rng_v = 77;
static float rf(void){ rng_v=rng_v*1103515245u+12345u; return(float)(rng_v&0x7FFF)/32768.0f; }
static float rn(void){ float u=rf(); if(u<1e-6f)u=1e-6f; return sqrtf(-2.0f*logf(u))*cosf(2.0f*M_PI*rf()); }

static float* data;

/* Generate diverse synthetic training data */
static void gen(float* img, int id) {
    int p = id % 24; /* More pattern variety */
    float ph = (float)(id / 24) * 0.03f;
    
    for (int y = 0; y < IMG_H; y++) for (int x = 0; x < IMG_W; x++) {
        int i = (y * IMG_W + x) * 3;
        float u = (float)x / (IMG_W - 1), v = (float)y / (IMG_H - 1), n = rn() * 0.02f;
        
        float r = 0.5f, g = 0.5f, b = 0.5f;
        
        switch (p % 24) {
            case 0: r = u + n; g = 0.5f + n; b = 1 - u + n; break;
            case 1: r = 0.3f + n; g = v + n; b = (1 - v) * 0.7f + n; break;
            case 2: r = ((x / 8 + y / 8) % 2) ? 0.8f : 0.2f + n; g = ((x / 8) % 2) ? 0.7f : 0.3f + n; b = u * v + n; break;
            case 3: { float r2 = sqrtf((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 2;
                    r = fminf(1, r2 + n); g = fminf(1, 1 - r2 + n); b = 0.5f + 0.5f * sinf(r2 * 6.28f) + n; break; }
            case 4: r = 0.5f + 0.5f * sinf(u * 12.56f + ph) + n; g = 0.5f + 0.5f * cosf(v * 12.56f + ph) + n;
                    b = 0.5f + 0.5f * sinf((u + v) * 6.28f) + n; break;
            case 5: { float d = (u + v) * 0.5f;
                    r = 0.5f + 0.5f * sinf(d * 12.56f + ph) + n; g = 0.5f + 0.5f * cosf(d * 18.84f + ph) + n;
                    b = 0.3f + 0.2f * sinf(d * 6.28f) + n; break; }
            case 6: { int b2 = (x < 4 || x >= IMG_W - 4 || y < 4 || y >= IMG_H - 4);
                    r = b2 ? 0.9f : 0.2f + n; g = b2 ? 0.1f : 0.6f + n; b = b2 ? 0.1f : 0.3f + n; break; }
            case 7: { float bx = u - 0.3f, by = v - 0.7f, g1 = expf(-(bx * bx + by * by) * 20);
                    bx = u - 0.7f; by = v - 0.3f; float g2 = expf(-(bx * bx + by * by) * 15);
                    r = g1 + n; g = g2 + n; b = (g1 + g2) * 0.5f + n; break; }
            case 8: { float a = atan2f(v - 0.5f, u - 0.5f);
                    r = 0.5f + 0.5f * sinf(a * 3 + ph) + n; g = 0.5f + 0.5f * cosf(a * 5 + ph) + n;
                    b = 0.5f + 0.5f * sinf(a * 7 + ph) + n; break; }
            case 9: { float c = ((x / 6) % 2 == (y / 6) % 2) ? 0.7f : 0.3f;
                    r = c * sinf(u * 12.56f + ph) + 0.5f + n; g = c * cosf(v * 12.56f + ph) + 0.5f + n;
                    b = c * sinf((u + v) * 8.88f) + 0.5f + n; break; }
            case 10: { float r2 = sqrtf((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f)) * 20;
                    r = 0.5f + 0.5f * sinf(r2 + ph) + n; g = 0.5f + 0.5f * cosf(r2 * 0.7f + ph) + n;
                    b = 0.5f + 0.5f * sinf(r2 * 1.3f + ph * 2) + n; break; }
            case 11: { int cx = (x > IMG_W / 2 - 8 && x < IMG_W / 2 + 8), cy = (y > IMG_H / 2 - 8 && y < IMG_H / 2 + 8);
                    r = (cx || cy) ? 0.8f : 0.2f + n; g = cx ? 0.1f : 0.6f + n; b = cy ? 0.1f : 0.6f + n; break; }
            /* New patterns for more diversity */
            case 12: { float ring = sinf(sqrtf((u-0.5f)*(u-0.5f) + (v-0.5f)*(v-0.5f)) * 20);
                    r = 0.5f + 0.5f * ring + n; g = 0.5f - 0.5f * ring + n; b = 0.5f + n; break; }
            case 13: { float diag = sinf((u + v) * 15.0f);
                    r = 0.5f + 0.5f * diag + n; g = 0.5f + 0.5f * sinf(diag * 3.14f) + n; b = 0.5f + 0.5f * cosf(diag * 2.0f) + n; break; }
            case 14: { float grid_x = sinf(u * 25.0f), grid_y = sinf(v * 25.0f);
                    r = (grid_x * grid_y + 1) * 0.5f + n; g = (grid_x + 1) * 0.5f + n; b = (grid_y + 1) * 0.5f + n; break; }
            case 15: { float spiral = sinf(atan2f(v-0.5f, u-0.5f) * 5 + sqrtf((u-0.5f)*(u-0.5f)+(v-0.5f)*(v-0.5f)) * 15);
                    r = 0.5f + 0.5f * spiral + n; g = 0.5f + 0.5f * cosf(spiral * 2) + n; b = 0.5f + 0.5f * sinf(spiral * 3) + n; break; }
            case 16: { float checker = ((x/4 + y/4) % 2) ? 0.7f : 0.3f;
                    r = checker * sinf(u * 20 + ph) + n; g = checker * cosf(v * 20 + ph) + n; b = checker * sinf((u+v) * 10) + n; break; }
            case 17: { float waves = sinf(u * 10 + ph) * cosf(v * 10 + ph);
                    r = 0.5f + 0.5f * waves + n; g = 0.5f + 0.5f * sinf(waves * 5) + n; b = 0.5f + 0.5f * cosf(waves * 7) + n; break; }
            case 18: { float radial = fmaxf(0, 1 - sqrtf((u-0.5f)*(u-0.5f) + (v-0.5f)*(v-0.5f)) * 3);
                    g = radial + n; r = radial * 0.7f + n; b = radial * 0.3f + n; break; }
            case 19: { float stripes = sinf((u + v) * 12);
                    r = 0.5f + 0.5f * stripes + n; g = 0.5f + 0.5f * sinf(stripes * 2) + n; b = 0.5f + 0.5f * cosf(stripes * 3) + n; break; }
            case 20: { float voronoi = sinf(u * 8) * cosf(v * 8);
                    r = 0.5f + 0.5f * voronoi + n; g = 0.5f + 0.5f * sinf(voronoi * 4) + n; b = 0.5f + 0.5f * cosf(voronoi * 6) + n; break; }
            case 21: { float diamond = fabsf(u - 0.5f) + fabsf(v - 0.5f);
                    r = 0.5f + 0.5f * sinf(diamond * 10) + n; g = 0.5f + 0.5f * cosf(diamond * 8) + n; b = 0.5f + 0.5f * sinf(diamond * 12) + n; break; }
            case 22: { float noise = rn() * 0.3f;
                    r = u + noise; g = v + noise; b = (u + v) * 0.5f + noise; break; }
            case 23: { float gradient = u * v;
                    r = gradient + n; g = (1 - u) * v + n; b = u * (1 - v) + n; break; }
        }
        
        img[i] = fminf(1, fmaxf(0, r));
        img[i+1] = fminf(1, fmaxf(0, g));
        img[i+2] = fminf(1, fmaxf(0, b));
    }
}

static inline float gelu_f(float x) { return 0.5f * x * (1 + tanhf(0.7978845608f * x * (1 + 0.044715f * x * x))); }
static inline float gelu_b(float x) { float c = 0.7978845608f, inner = c * x * (1 + 0.044715f * x * x), t = tanhf(inner);
    return 0.5f * (1 + t) + 0.5f * x * (1 - t * t) * c * (1 + 3 * 0.044715f * x * x); }

/* Deeper model */
typedef struct {
    float pw[PP*DIM], pb[DIM], plg[DIM], plb[DIM];
    float fw1[LVL][DIM*FH], fb1[LVL][FH];
    float fw2[LVL][FH*DIM], fb2[LVL][DIM];
    /* Deeper decoder: 2 layers */
    float dw1[LDIM*FH], db1[FH];
    float dw2[FH*PP], db2[PP];
    /* Adam moments */
    float pwm[PP*DIM], pwv[PP*DIM], pbm[DIM], pbv[DIM], plgm[DIM], plgv[DIM], plbm[DIM], plbv[DIM];
    float fwm[LVL][DIM*FH], fwv[LVL][DIM*FH], fbm[LVL][FH], fbv[LVL][FH];
    float fwm2[LVL][FH*DIM], fwv2[LVL][FH*DIM], fbm2[LVL][DIM], fbv2[LVL][DIM];
    float dw1m[LDIM*FH], dw1v[LDIM*FH], db1m[FH], db1v[FH];
    float dw2m[FH*PP], dw2v[FH*PP], db2m[PP], db2v[PP];
    int step;
} Model;

static Model model;

static void init_model(void) {
    memset(&model, 0, sizeof(model));
    for (int i = 0; i < DIM; i++) model.plg[i] = 1;
    float lim = sqrtf(6.0 / (PP + DIM));
    for (int i = 0; i < PP * DIM; i++) model.pw[i] = (rf() * 2 - 1) * lim;
    for (int i = 0; i < LDIM * FH; i++) model.dw1[i] = rn() * 0.01f;
    for (int i = 0; i < FH * PP; i++) model.dw2[i] = rn() * 0.01f;
    for (int l = 0; l < LVL; l++) {
        for (int h = 0; h < FH; h++) for (int d = 0; d < DIM; d++) model.fw1[l][h * DIM + d] = rn() * 0.01f;
    }
    model.step = 0;
}

/* Forward + backward */
static float forward_backward(float* img,
                               float* gpw, float* gpb, float* gplg, float* gplb,
                               float* gfw1, float* gfb1, float* gfw2, float* gfb2,
                               float* gdw1, float* gdb1, float* gdw2, float* gdb2) {
    float loss = 0;

    for (int p = 0; p < NP; p++) {
        float pd[PP];
        int py = p / NPW, px = p % NPW, idx = 0;
        for (int dy = 0; dy < PS; dy++) for (int dx = 0; dx < PS; dx++) {
            int iy = py * PS + dy, ix = px * PS + dx, ii = (iy * IMG_W + ix) * 3;
            pd[idx++] = img[ii]; pd[idx++] = img[ii + 1]; pd[idx++] = img[ii + 2];
        }

        /* Encoder */
        float pre[DIM];
        for (int o = 0; o < DIM; o++) {
            float s = model.pb[o];
            for (int i = 0; i < PP; i++) s += model.pw[o * PP + i] * pd[i];
            pre[o] = s;
        }

        float mean = 0; for (int i = 0; i < DIM; i++) mean += pre[i]; mean /= DIM;
        float var = 0; for (int i = 0; i < DIM; i++) { float d = pre[i] - mean; var += d * d; } var /= DIM;
        float inv = 1 / sqrtf(var + 1e-5f);
        float pl[DIM];
        for (int i = 0; i < DIM; i++) pl[i] = model.plg[i] * (pre[i] - mean) * inv + model.plb[i];

        float v[DIM];
        memcpy(v, pl, DIM * sizeof(float));

        float v_in[LVL][DIM], hb_store[LVL][FH], hba_store[LVL][FH];

        for (int l = 0; l < LVL; l++) {
            memcpy(v_in[l], v, DIM * sizeof(float));
            float hb[FH], hba[FH], fv[DIM];
            for (int h = 0; h < FH; h++) {
                float s = model.fb1[l][h];
                for (int i = 0; i < DIM; i++) s += model.fw1[l][h * DIM + i] * v[i];
                hb[h] = s; hba[h] = gelu_f(s);
            }
            for (int o = 0; o < DIM; o++) {
                float s = model.fb2[l][o];
                for (int h = 0; h < FH; h++) s += model.fw2[l][o * FH + h] * hba[h];
                fv[o] = s;
            }
            for (int i = 0; i < DIM; i++) v[i] += fv[i];
            memcpy(hb_store[l], hb, sizeof(hb));
            memcpy(hba_store[l], hba, sizeof(hba));
        }

        float lat[LDIM];
        for (int i = 0; i < LDIM; i++) lat[i] = v[i];

        /* Deeper decoder: 2 layers with GELU */
        float h1[FH], h1a[FH];
        for (int h = 0; h < FH; h++) {
            float s = model.db1[h];
            for (int i = 0; i < LDIM; i++) s += model.dw1[h * LDIM + i] * lat[i];
            h1[h] = s; h1a[h] = gelu_f(s);
        }

        float dlat[LDIM];
        memset(dlat, 0, sizeof(dlat));
        for (int o = 0; o < PP; o++) {
            float pred = model.db2[o];
            for (int h = 0; h < FH; h++) pred += model.dw2[o * FH + h] * h1a[h];
            float gt = pd[o];
            float d = pred - gt;
            loss += d * d;
            float grad = 2.0f * d / (float)(NP * PP);
            gdw2[o] += grad;
            for (int h = 0; h < FH; h++) {
                gdw2[o * FH + h] += grad * h1a[h];
                dlat[h] += grad * model.dw2[o * FH + h];
            }
        }

        /* Decoder backward */
        float dh1[FH];
        for (int h = 0; h < FH; h++) dh1[h] = dlat[h] * gelu_b(h1[h]);
        for (int h = 0; h < FH; h++) {
            float g = dh1[h];
            gdb1[h] += g;
            for (int i = 0; i < LDIM; i++) {
                gdw1[h * LDIM + i] += g * lat[i];
                dlat[i] += g * model.dw1[h * LDIM + i];
            }
        }

        /* Encoder backward */
        float dv[DIM];
        for (int i = 0; i < DIM && i < LDIM; i++) dv[i] = dlat[i];
        for (int i = LDIM; i < DIM; i++) dv[i] = 0;

        for (int l = LVL - 1; l >= 0; l--) {
            float dfv[DIM];
            for (int i = 0; i < DIM; i++) dfv[i] = dv[i];

            float dhba[FH];
            memset(dhba, 0, sizeof(dhba));
            for (int o = 0; o < DIM; o++) {
                float g = dfv[o];
                gfb2[o] += g;
                for (int h = 0; h < FH; h++) {
                    gfw2[o * FH + h] += g * hba_store[l][h];
                    dhba[h] += g * model.fw2[l][o * FH + h];
                }
            }

            float dhb[FH];
            for (int h = 0; h < FH; h++) dhb[h] = dhba[h] * gelu_b(hb_store[l][h]);

            for (int h = 0; h < FH; h++) {
                float g = dhb[h];
                gfb1[h] += g;
                for (int i = 0; i < DIM; i++) {
                    gfw1[h * DIM + i] += g * v_in[l][i];
                    dv[i] += g * model.fw1[l][h * DIM + i];
                }
            }
        }

        /* LayerNorm backward */
        float dvar = 0;
        for (int i = 0; i < DIM; i++) {
            float dxhat = dv[i] * model.plg[i];
            dvar += dxhat * (pre[i] - mean);
        }
        dvar *= -0.5f * powf(var, -1.5f);

        for (int i = 0; i < DIM; i++) {
            float dxhat = dv[i] * model.plg[i];
            float dx = dxhat * inv;
            dx += dxhat * (-0.5f) * inv * inv * inv * 2.0f * (pre[i] - mean) * dvar;
            float dmean = 0;
            for (int j = 0; j < DIM; j++) {
                float dyhat = dv[j] * model.plg[j];
                dmean += -dyhat * inv;
                dmean += -dyhat * inv * 2.0f * (pre[j] - mean) * dvar;
            }
            dmean /= (float)DIM;
            dx += dmean;

            gplg[i] += (pre[i] - mean) * inv * dv[i];
            gplb[i] += dv[i];
            gpb[i] += dx;
            for (int j = 0; j < PP; j++) gpw[i * PP + j] += dx * pd[j];
        }
    }

    return loss / (float)(NP * PP);
}

int main(void) {
    printf("WuBuMath Deep Model Training\n");
    printf("Deeper architecture: %d levels, %d hidden, %d flow hidden\n", LVL, DIM, FH);
    printf("Deeper decoder: 2 layers with GELU\n\n");
    
    init_model();

    printf("Generating %d training samples...\n", NSAMP);
    data = (float *)malloc((size_t)(NSAMP * IMG_H * IMG_W * 3) * sizeof(float));
    for (int n = 0; n < NSAMP; n++) gen(data + n * IMG_H * IMG_W * 3, n);
    printf("Dataset ready: %.1f MB\n\n", (float)(NSAMP * IMG_H * IMG_W * 3 * 4) / (1024 * 1024));

    float *gpw = calloc(PP * DIM, sizeof(float)), *gpb = calloc(DIM, sizeof(float));
    float *gplg = calloc(DIM, sizeof(float)), *gplb = calloc(DIM, sizeof(float));
    float *gfw1 = calloc(DIM * FH, sizeof(float)), *gfb1 = calloc(FH, sizeof(float));
    float *gfw2 = calloc(FH * DIM, sizeof(float)), *gfb2 = calloc(DIM, sizeof(float));
    float *gdw1 = calloc(LDIM * FH, sizeof(float)), *gdb1 = calloc(FH, sizeof(float));
    float *gdw2 = calloc(FH * PP, sizeof(float)), *gdb2 = calloc(PP, sizeof(float));

    FILE *log = fopen("/home/wubu/WuBuMath/output/train_log_deep.csv", "w");
    fprintf(log, "epoch,loss,psnr\n");

    printf("Training: %d epochs, batch %d\n", EPOCHS, BATCH);
    printf("Image: %dx%d, patches: %d, latent: %d bytes\n", IMG_H, IMG_W, NP, LDIM);
    printf("Compression: %.0f:1\n\n", (float)(IMG_H * IMG_W * 3 * 4) / (float)LDIM);

    for (int ep = 0; ep < EPOCHS; ep++) {
        float el = 0;
        int nb = NSAMP / BATCH;
        for (int b = 0; b < nb; b++) {
            int s = b * BATCH;
            float bl = 0;
            memset(gpw, 0, sizeof(float) * PP * DIM); memset(gpb, 0, sizeof(float) * DIM);
            memset(gplg, 0, sizeof(float) * DIM); memset(gplb, 0, sizeof(float) * DIM);
            memset(gfw1, 0, sizeof(float) * DIM * FH); memset(gfb1, 0, sizeof(float) * FH);
            memset(gfw2, 0, sizeof(float) * FH * DIM); memset(gfb2, 0, sizeof(float) * DIM);
            memset(gdw1, 0, sizeof(float) * LDIM * FH); memset(gdb1, 0, sizeof(float) * FH);
            memset(gdw2, 0, sizeof(float) * FH * PP); memset(gdb2, 0, sizeof(float) * PP);
            for (int n = s; n < s + BATCH; n++)
                bl += forward_backward(data + n * IMG_H * IMG_W * 3,
                    gpw, gpb, gplg, gplb, gfw1, gfb1, gfw2, gfb2, gdw1, gdb1, gdw2, gdb2);
            bl /= BATCH; el += bl; model.step++;

            float lr = 3e-4f * (1.0f - (float)ep / (float)EPOCHS * 0.5f);
            float bc1 = 1 - powf(.9f, model.step), bc2 = 1 - powf(.999f, model.step);

            /* Adam updates */
            for (int i = 0; i < PP * DIM; i++) {
                float gi = gpw[i] / BATCH;
                model.pwm[i] = .9f * model.pwm[i] + .1f * gi;
                model.pwv[i] = .999f * model.pwv[i] + .001f * gi * gi;
                model.pw[i] -= lr * (model.pwm[i] / bc1) / (sqrtf(model.pwv[i] / bc2) + 1e-8f);
            }
            for (int i = 0; i < DIM; i++) {
                float gi = gpb[i] / BATCH;
                model.pbm[i] = .9f * model.pbm[i] + .1f * gi;
                model.pbv[i] = .999f * model.pbv[i] + .001f * gi * gi;
                model.pb[i] -= lr * (model.pbm[i] / bc1) / (sqrtf(model.pbv[i] / bc2) + 1e-8f);
                model.plgm[i] = .9f * model.plgm[i] + .1f * (gplg[i] / BATCH);
                model.plgv[i] = .999f * model.plgv[i] + .001f * (gplg[i] / BATCH) * (gplg[i] / BATCH);
                model.plg[i] -= lr * (model.plgm[i] / bc1) / (sqrtf(model.plgv[i] / bc2) + 1e-8f);
                model.plbm[i] = .9f * model.plbm[i] + .1f * (gplb[i] / BATCH);
                model.plbv[i] = .999f * model.plbv[i] + .001f * (gplb[i] / BATCH) * (gplb[i] / BATCH);
                model.plb[i] -= lr * (model.plbm[i] / bc1) / (sqrtf(model.plbv[i] / bc2) + 1e-8f);
            }
            for (int i = 0; i < LDIM * FH; i++) {
                float gi = gdw1[i] / BATCH;
                model.dw1m[i] = .9f * model.dw1m[i] + .1f * gi;
                model.dw1v[i] = .999f * model.dw1v[i] + .001f * gi * gi;
                model.dw1[i] -= lr * (model.dw1m[i] / bc1) / (sqrtf(model.dw1v[i] / bc2) + 1e-8f);
            }
            for (int i = 0; i < FH; i++) {
                float gi = gdb1[i] / BATCH;
                model.db1m[i] = .9f * model.db1m[i] + .1f * gi;
                model.db1v[i] = .999f * model.db1v[i] + .001f * gi * gi;
                model.db1[i] -= lr * (model.db1m[i] / bc1) / (sqrtf(model.db1v[i] / bc2) + 1e-8f);
            }
            for (int i = 0; i < FH * PP; i++) {
                float gi = gdw2[i] / BATCH;
                model.dw2m[i] = .9f * model.dw2m[i] + .1f * gi;
                model.dw2v[i] = .999f * model.dw2v[i] + .001f * gi * gi;
                model.dw2[i] -= lr * (model.dw2m[i] / bc1) / (sqrtf(model.dw2v[i] / bc2) + 1e-8f);
            }
            for (int i = 0; i < PP; i++) {
                float gi = gdb2[i] / BATCH;
                model.db2m[i] = .9f * model.db2m[i] + .1f * gi;
                model.db2v[i] = .999f * model.db2v[i] + .001f * gi * gi;
                model.db2[i] -= lr * (model.db2m[i] / bc1) / (sqrtf(model.db2v[i] / bc2) + 1e-8f);
            }
            for (int l = 0; l < LVL; l++) {
                for (int i = 0; i < DIM * FH; i++) {
                    float gi = gfw1[i] / BATCH;
                    model.fwm[l][i] = .9f * model.fwm[l][i] + .1f * gi;
                    model.fwv[l][i] = .999f * model.fwv[l][i] + .001f * gi * gi;
                    model.fw1[l][i] -= lr * (model.fwm[l][i] / bc1) / (sqrtf(model.fwv[l][i] / bc2) + 1e-8f);
                }
                for (int i = 0; i < FH; i++) {
                    float gi = gfb1[i] / BATCH;
                    model.fbm[l][i] = .9f * model.fbm[l][i] + .1f * gi;
                    model.fbv[l][i] = .999f * model.fbv[l][i] + .001f * gi * gi;
                    model.fb1[l][i] -= lr * (model.fbm[l][i] / bc1) / (sqrtf(model.fbv[l][i] / bc2) + 1e-8f);
                }
                for (int i = 0; i < FH * DIM; i++) {
                    float gi = gfw2[i] / BATCH;
                    model.fwm2[l][i] = .9f * model.fwm2[l][i] + .1f * gi;
                    model.fwv2[l][i] = .999f * model.fwv2[l][i] + .001f * gi * gi;
                    model.fw2[l][i] -= lr * (model.fwm2[l][i] / bc1) / (sqrtf(model.fwv2[l][i] / bc2) + 1e-8f);
                }
                for (int i = 0; i < DIM; i++) {
                    float gi = gfb2[i] / BATCH;
                    model.fbm2[l][i] = .9f * model.fbm2[l][i] + .1f * gi;
                    model.fbv2[l][i] = .999f * model.fbv2[l][i] + .001f * gi * gi;
                    model.fb2[l][i] -= lr * (model.fbm2[l][i] / bc1) / (sqrtf(model.fbv2[l][i] / bc2) + 1e-8f);
                }
            }
        }
        el /= nb;
        float psnr = el < 1e-10 ? 100 : 10 * log10f(1 / el);
        printf("Epoch %3d/%d | Loss: %.6f | PSNR: %.1f dB\n", ep + 1, EPOCHS, el, psnr);
        fprintf(log, "%d,%.6f,%.2f\n", ep + 1, el, psnr);
        if ((ep + 1) % 50 == 0) {
            char path[256]; snprintf(path, sizeof(path), "/home/wubu/WuBuMath/output/ckpt_deep_e%03d.bin", ep + 1);
            FILE *f = fopen(path, "wb"); fwrite(&model, sizeof(Model), 1, f); fclose(f);
            printf("  -> Saved: %s\n", path);
        }
    }
    fclose(log);

    FILE *f = fopen("/home/wubu/WuBuMath/output/wubu_deep_final.bin", "wb");
    fwrite(&model, sizeof(Model), 1, f); fclose(f);
    printf("\nDone. Model: output/wubu_deep_final.bin\n");
    printf("Log: output/train_log_deep.csv\n");

    free(data);
    return 0;
}
