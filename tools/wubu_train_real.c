/* wubu_train_real.c — WuBuMath Training on Real Video Frames
 *
 * Trains directly on DBZ frames instead of synthetic data.
 * The model learns the actual content distribution.
 * 
 * Pipeline:
 * 1. Load DBZ frames (downsampled to 64x64)
 * 2. Train encoder-decoder with backpropagation
 * 3. Save trained model
 * 4. Use for compression
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

/* Match training parameters from train_wubu_codec.c */
#define IMG_H 64
#define IMG_W 64
#define PS 4
#define NPH 16
#define NPW 16
#define NP 256
#define PP 48
#define DIM 32
#define LDIM 32
#define LVL 4
#define FH 32

#define BATCH 16
#define EPOCHS 200

static unsigned rng_v = 77;
static float rf(void){ rng_v=rng_v*1103515245u+12345u; return(float)(rng_v&0x7FFF)/32768.0f; }
static float rn(void){ float u=rf(); if(u<1e-6f)u=1e-6f; return sqrtf(-2.0f*logf(u))*cosf(2.0f*M_PI*rf()); }

static float* data;
static int num_samples = 0;

/* Load real DBZ frames and downsample to 64x64 */
static int load_real_data(const char *path, int W, int H) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    int nf = (int)(fsz / (W * H * 3));
    if (nf == 0) { fclose(f); return 0; }
    
    /* Allocate for all frames */
    data = (float *)malloc((size_t)nf * IMG_H * IMG_W * 3 * sizeof(float));
    
    uint8_t *frame = (uint8_t *)malloc(W * H * 3);
    num_samples = 0;
    
    for (int i = 0; i < nf && i < 512; i++) {
        if (fread(frame, 1, W * H * 3, f) != (size_t)(W * H * 3)) break;
        
        /* Downsample to 64x64 and normalize to [0,1] */
        for (int y = 0; y < IMG_H; y++) {
            for (int x = 0; x < IMG_W; x++) {
                int sy = y * H / IMG_H;
                int sx = x * W / IMG_W;
                int src_idx = (sy * W + sx) * 3;
                int dst_idx = (y * IMG_W + x) * 3;
                
                data[num_samples * IMG_H * IMG_W * 3 + dst_idx + 0] = frame[src_idx + 0] / 255.0f;
                data[num_samples * IMG_H * IMG_W * 3 + dst_idx + 1] = frame[src_idx + 1] / 255.0f;
                data[num_samples * IMG_H * IMG_W * 3 + dst_idx + 2] = frame[src_idx + 2] / 255.0f;
            }
        }
        num_samples++;
    }
    
    free(frame);
    fclose(f);
    printf("Loaded %d frames (%dx%d → %dx%d)\n", num_samples, W, H, IMG_W, IMG_H);
    return num_samples;
}

static inline float gelu_f(float x) { return 0.5f * x * (1 + tanhf(0.7978845608f * x * (1 + 0.044715f * x * x))); }
static inline float gelu_b(float x) { float c = 0.7978845608f, inner = c * x * (1 + 0.044715f * x * x), t = tanhf(inner);
    return 0.5f * (1 + t) + 0.5f * x * (1 - t * t) * c * (1 + 3 * 0.044715f * x * x); }

typedef struct {
    float pw[PP*DIM], pb[DIM], plg[DIM], plb[DIM];
    float fw1[LVL][DIM*FH], fb1[LVL][FH];
    float fw2[LVL][FH*DIM], fb2[LVL][DIM];
    float dw[NP*PP*LDIM], db[NP*PP];
    /* Adam moments */
    float pwm[PP*DIM], pwv[PP*DIM], pbm[DIM], pbv[DIM], plgm[DIM], plgv[DIM], plbm[DIM], plbv[DIM];
    float fwm[LVL][DIM*FH], fwv[LVL][DIM*FH], fbm[LVL][FH], fbv[LVL][FH];
    float fwm2[LVL][FH*DIM], fwv2[LVL][FH*DIM], fbm2[LVL][DIM], fbv2[LVL][DIM];
    float dwm[NP*PP*LDIM], dwv[NP*PP*LDIM], dbm[NP*PP], dbv[NP*PP];
    int step;
} Model;

static Model model;

static void init_model(void) {
    memset(&model, 0, sizeof(model));
    for (int i = 0; i < DIM; i++) model.plg[i] = 1;
    float lim = sqrtf(6.0 / (PP + DIM));
    for (int i = 0; i < PP * DIM; i++) model.pw[i] = (rf() * 2 - 1) * lim;
    for (int i = 0; i < NP * PP * LDIM; i++) model.dw[i] = rn() * 0.005f;
    for (int l = 0; l < LVL; l++) {
        for (int h = 0; h < FH; h++) for (int d = 0; d < DIM; d++) model.fw1[l][h * DIM + d] = rn() * 0.01f;
    }
    model.step = 0;
}

/* Forward + backward for one sample */
static float forward_backward(float* img,
                               float* gpw, float* gpb, float* gplg, float* gplb,
                               float* gfw1, float* gfb1, float* gfw2, float* gfb2,
                               float* gdw, float* gdb) {
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

        /* Decoder + loss */
        float dlat[LDIM];
        memset(dlat, 0, sizeof(dlat));
        for (int o = 0; o < PP; o++) {
            float pred = model.db[p * PP + o];
            for (int i = 0; i < LDIM; i++) pred += model.dw[p * PP * LDIM + o * LDIM + i] * lat[i];
            float gt = pd[o];
            float d = pred - gt;
            loss += d * d;
            float grad = 2.0f * d / (float)(NP * PP);
            gdb[p * PP + o] += grad;
            for (int i = 0; i < LDIM; i++) {
                gdw[p * PP * LDIM + o * LDIM + i] += grad * lat[i];
                dlat[i] += grad * model.dw[p * PP * LDIM + o * LDIM + i];
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

int main(int argc, char **argv) {
    if (argc < 2) { printf("Usage: %s input.raw [W H]\n", argv[0]); return 1; }
    
    int W = 320, H = 240;
    if (argc >= 5) { W = atoi(argv[2]); H = atoi(argv[3]); }
    
    printf("WuBuMath Training on Real Video\n");
    printf("================================\n\n");
    
    /* Load real video frames */
    int ns = load_real_data(argv[1], W, H);
    if (ns == 0) { printf("No data loaded\n"); return 1; }
    
    printf("Training samples: %d\n", ns);
    printf("Image: %dx%d, patches: %d, latent: %d bytes\n", IMG_H, IMG_W, NP, LDIM);
    printf("Compression: %.0f:1\n\n", (float)(IMG_H * IMG_W * 3 * 4) / (float)LDIM);
    
    init_model();

    float *gpw = calloc(PP * DIM, sizeof(float)), *gpb = calloc(DIM, sizeof(float));
    float *gplg = calloc(DIM, sizeof(float)), *gplb = calloc(DIM, sizeof(float));
    float *gfw1 = calloc(DIM * FH, sizeof(float)), *gfb1 = calloc(FH, sizeof(float));
    float *gfw2 = calloc(FH * DIM, sizeof(float)), *gfb2 = calloc(DIM, sizeof(float));
    float *gdw = calloc(NP * PP * LDIM, sizeof(float)), *gdb = calloc(NP * PP, sizeof(float));

    FILE *log = fopen("/home/wubu/WuBuMath/output/train_log_real.csv", "w");
    fprintf(log, "epoch,loss,psnr\n");

    printf("Training: %d epochs, batch %d\n", EPOCHS, BATCH);
    
    for (int ep = 0; ep < EPOCHS; ep++) {
        float el = 0;
        int nb = ns / BATCH;
        for (int b = 0; b < nb; b++) {
            int s = b * BATCH;
            float bl = 0;
            memset(gpw, 0, sizeof(float) * PP * DIM); memset(gpb, 0, sizeof(float) * DIM);
            memset(gplg, 0, sizeof(float) * DIM); memset(gplb, 0, sizeof(float) * DIM);
            memset(gfw1, 0, sizeof(float) * DIM * FH); memset(gfb1, 0, sizeof(float) * FH);
            memset(gfw2, 0, sizeof(float) * FH * DIM); memset(gfb2, 0, sizeof(float) * DIM);
            memset(gdw, 0, sizeof(float) * NP * PP * LDIM); memset(gdb, 0, sizeof(float) * NP * PP);
            for (int n = s; n < s + BATCH; n++)
                bl += forward_backward(data + n * IMG_H * IMG_W * 3,
                    gpw, gpb, gplg, gplb, gfw1, gfb1, gfw2, gfb2, gdw, gdb);
            bl /= BATCH; el += bl; model.step++;

            float lr = 3e-4f * (1.0f - (float)ep / (float)EPOCHS * 0.5f);
            float bc1 = 1 - powf(.9f, model.step), bc2 = 1 - powf(.999f, model.step);

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
            for (int i = 0; i < NP * PP * LDIM; i++) {
                float gi = gdw[i] / BATCH;
                model.dwm[i] = .9f * model.dwm[i] + .1f * gi;
                model.dwv[i] = .999f * model.dwv[i] + .001f * gi * gi;
                model.dw[i] -= lr * (model.dwm[i] / bc1) / (sqrtf(model.dwv[i] / bc2) + 1e-8f);
            }
            for (int i = 0; i < NP * PP; i++) {
                float gi = gdb[i] / BATCH;
                model.dbm[i] = .9f * model.dbm[i] + .1f * gi;
                model.dbv[i] = .999f * model.dbv[i] + .001f * gi * gi;
                model.db[i] -= lr * (model.dbm[i] / bc1) / (sqrtf(model.dbv[i] / bc2) + 1e-8f);
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
            char path[256]; snprintf(path, sizeof(path), "/home/wubu/WuBuMath/output/ckpt_real_e%03d.bin", ep + 1);
            FILE *f = fopen(path, "wb"); fwrite(&model, sizeof(Model), 1, f); fclose(f);
            printf("  -> Saved: %s\n", path);
        }
    }
    fclose(log);

    FILE *f = fopen("/home/wubu/WuBuMath/output/wubu_real_final.bin", "wb");
    fwrite(&model, sizeof(Model), 1, f); fclose(f);
    printf("\nDone. Model: output/wubu_real_final.bin\n");
    printf("Log: output/train_log_real.csv\n");

    free(data);
    return 0;
}
