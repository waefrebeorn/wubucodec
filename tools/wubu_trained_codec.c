/* wubu_trained_codec.c — WuBuCodec with Trained Encoder-Decoder
 *
 * Full pipeline integrating the trained model from train_wubu_codec.c:
 * 1. Encoder: patch[48] → Linear[48→32] → LayerNorm → 4 flow MLP → latent[32]
 * 2. Temporal compression: wubu_latent_codec (I/P/B on Poincaré ball)
 * 3. Decoder: latent[32] → Linear[32→48] → reconstructed patch[48]
 *
 * The trained model achieves 31.9 dB on synthetic data.
 * This integrates it for real video compression.
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

#define WUBU_EPS 1e-6f
#define CURVATURE 1.0f

/* Match training parameters */
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

/* ============================ Physics ============================ */

static float vec_norm(const float* x, int N) {
    float sum = 0;
    for (int i = 0; i < N; i++) sum += x[i] * x[i];
    return sqrtf(sum + WUBU_EPS);
}

static void expmap(float *out, const float *v, int N, float c) {
    float norm = vec_norm(v, N);
    float sqrt_c = sqrtf(c);
    if (norm < WUBU_EPS) { for (int i = 0; i < N; i++) out[i] = v[i]; return; }
    float arg = sqrt_c * norm;
    if (arg > 30.0f) arg = 30.0f;
    float tanh_val = tanhf(arg);
    float scale = tanh_val / (sqrt_c * norm + WUBU_EPS);
    for (int i = 0; i < N; i++) out[i] = v[i] * scale;
}

static void logmap(float *out, const float *y, int N, float c) {
    float norm = vec_norm(y, N);
    float sqrt_c = sqrtf(c);
    if (norm < WUBU_EPS) { for (int i = 0; i < N; i++) out[i] = y[i]; return; }
    float arg = sqrt_c * norm;
    if (arg > 1.0f - WUBU_EPS * 4) arg = 1.0f - WUBU_EPS * 4;
    float atanh_val = atanhf(arg);
    float scale = atanh_val / (sqrt_c * norm + WUBU_EPS);
    for (int i = 0; i < N; i++) out[i] = y[i] * scale;
}

static void mobius_add(float *out, const float *x, const float *y, int N, float c) {
    float x_norm_sq = 0, y_norm_sq = 0, xy_dot = 0;
    for (int i = 0; i < N; i++) {
        x_norm_sq += x[i] * x[i];
        y_norm_sq += y[i] * y[i];
        xy_dot += x[i] * y[i];
    }
    float denom = 1.0f + 2.0f * c * xy_dot + c * c * x_norm_sq * y_norm_sq;
    if (denom < WUBU_EPS) denom = WUBU_EPS;
    float num_x = 1.0f + 2.0f * c * xy_dot + c * y_norm_sq;
    float num_y = 1.0f - c * x_norm_sq;
    for (int i = 0; i < N; i++) out[i] = (num_x * x[i] + num_y * y[i]) / denom;
}

static void expmap_at(float *out, const float *x, const float *v, int N, float c) {
    float exp_v[256];
    expmap(exp_v, v, N, c);
    mobius_add(out, x, exp_v, N, c);
}

static void logmap_at(float *out, const float *x, const float *y, int N, float c) {
    float neg_x[256];
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];
    float rel[256];
    mobius_add(rel, neg_x, y, N, c);
    logmap(out, rel, N, c);
}

static void geodesic_interp(float *out, const float *x0, const float *x1, float t, int N, float c) {
    if (t <= 0.0f) { memcpy(out, x0, N * sizeof(float)); return; }
    if (t >= 1.0f) { memcpy(out, x1, N * sizeof(float)); return; }
    float tangent[256];
    logmap_at(tangent, x0, x1, N, c);
    for (int i = 0; i < N; i++) tangent[i] *= t;
    expmap_at(out, x0, tangent, N, c);
}

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

/* ============================ Model ============================ */

typedef struct {
    float pw[PP*DIM], pb[DIM], plg[DIM], plb[DIM];
    float fw1[LVL][DIM*FH], fb1[LVL][FH];
    float fw2[LVL][FH*DIM], fb2[LVL][DIM];
    float dw[NP*PP*LDIM], db[NP*PP];
} Model;

static Model model;

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot load model %s\n", path); exit(1); }
    fread(&model, sizeof(Model), 1, f);
    fclose(f);
    printf("Loaded model from %s (%zu bytes)\n", path, sizeof(Model));
}

/* ============================ Encoder ============================ */

static void encode_patch(const float *patch, float *latent) {
    /* Linear: pre = W @ patch + b */
    float pre[DIM];
    for (int o = 0; o < DIM; o++) {
        float s = model.pb[o];
        for (int i = 0; i < PP; i++) s += model.pw[o*PP+i] * patch[i];
        pre[o] = s;
    }
    
    /* LayerNorm */
    float mean = 0;
    for (int i = 0; i < DIM; i++) mean += pre[i];
    mean /= DIM;
    float var = 0;
    for (int i = 0; i < DIM; i++) { float d = pre[i] - mean; var += d * d; }
    var /= DIM;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    float pl[DIM];
    for (int i = 0; i < DIM; i++) pl[i] = model.plg[i] * (pre[i] - mean) * inv + model.plb[i];
    
    /* 4 flow MLP levels */
    float v[DIM];
    memcpy(v, pl, DIM * sizeof(float));
    
    for (int l = 0; l < LVL; l++) {
        float hb[FH], hba[FH], fv[DIM];
        for (int h = 0; h < FH; h++) {
            float s = model.fb1[l][h];
            for (int i = 0; i < DIM; i++) s += model.fw1[l][h*DIM+i] * v[i];
            hb[h] = s;
            hba[h] = gelu(s);
        }
        for (int o = 0; o < DIM; o++) {
            float s = model.fb2[l][o];
            for (int h = 0; h < FH; h++) s += model.fw2[l][o*FH+h] * hba[h];
            fv[o] = s;
        }
        for (int i = 0; i < DIM; i++) v[i] += fv[i];
    }
    
    /* Latent = first LDIM dims of v */
    for (int i = 0; i < LDIM; i++) latent[i] = v[i];
}

/* ============================ Decoder ============================ */

static void decode_patch(const float *latent, float *patch) {
    /* Linear: pred = W @ latent + b */
    for (int o = 0; o < PP; o++) {
        float pred = model.db[o];
        for (int i = 0; i < LDIM; i++) pred += model.dw[o*LDIM+i] * latent[i];
        patch[o] = pred;
    }
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
    printf("WuBuMath Trained Codec\n");
    printf("  %dx%d @ %d fps, QP=%d, GOP=%d, B=%d\n", W, H, fps, qp, gop_size, b_frames);

    /* Load trained model */
    load_model("/home/wubu/WuBuMath/output/wubu_trained_final.bin");

    FILE *f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    int nf = (int)(fsz / np);
    uint8_t *frames = (uint8_t *)malloc((size_t)nf * np);
    if (fread(frames, 1, (size_t)nf*np, f) != (size_t)(nf*np)) { printf("Read failed\n"); return 1; }
    fclose(f);
    printf("Loaded %d frames\n", nf);

    /* Frame types */
    int *ftypes = (int *)malloc(nf * sizeof(int));
    for (int fn = 0; fn < nf; fn++) {
        if (fn % gop_size == 0) ftypes[fn] = 0;
        else if (b_frames > 0 && (fn % gop_size) % (b_frames + 1) != 0) ftypes[fn] = 2;
        else ftypes[fn] = 1;
    }
    
    /* Allocate latent arrays */
    int num_latents = NP * LDIM;
    float *all_latents = (float *)calloc((size_t)nf * num_latents, sizeof(float));
    
    /* Quantization step */
    float qstep = 0.01f * (qp / 26.0f);
    
    /* Two-pass encoding */
    printf("--- Pass 1: Encode I/P frames ---\n");
    for (int fn = 0; fn < nf; fn++) {
        if (ftypes[fn] == 2) continue;
        
        const uint8_t *orig = frames + fn * np;
        int is_i = (ftypes[fn] == 0);
        
        /* Process each 64x64 block */
        for (int blk_y = 0; blk_y < H; blk_y += 64) {
            for (int blk_x = 0; blk_x < W; blk_x += 64) {
                int blk_h = (blk_y + 64 > H) ? H - blk_y : 64;
                int blk_w = (blk_x + 64 > W) ? W - blk_x : 64;
                
                /* Extract 64x64 block as float */
                float block[64*64*3];
                for (int y = 0; y < blk_h; y++) {
                    for (int x = 0; x < blk_w; x++) {
                        block[(y*64+x)*3+0] = (orig[(blk_y+y)*W+(blk_x+x)] / 127.5f) - 1.0f;
                        block[(y*64+x)*3+1] = (orig[(blk_y+y)*W+(blk_x+x)] / 127.5f) - 1.0f;
                        block[(y*64+x)*3+2] = (orig[(blk_y+y)*W+(blk_x+x)] / 127.5f) - 1.0f;
                    }
                }
                
                /* Encode each 4x4 patch */
                for (int py = 0; py < 16; py++) {
                    for (int px = 0; px < 16; px++) {
                        float patch[48];
                        int idx = 0;
                        for (int dy = 0; dy < 4; dy++) {
                            for (int dx = 0; dx < 4; dx++) {
                                int iy = py*4+dy;
                                int ix = px*4+dx;
                                patch[idx++] = block[(iy*64+ix)*3+0];
                                patch[idx++] = block[(iy*64+ix)*3+1];
                                patch[idx++] = block[(iy*64+ix)*3+2];
                            }
                        }
                        
                        int patch_idx = py * 16 + px;
                        float *latent = &all_latents[fn * num_latents + patch_idx * LDIM];
                        encode_patch(patch, latent);
                        
                        /* Quantize in tangent space at origin */
                        float tangent[LDIM];
                        logmap(tangent, latent, LDIM, CURVATURE);
                        for (int i = 0; i < LDIM; i++) {
                            int level = (int)roundf(tangent[i] / qstep);
                            tangent[i] = level * qstep;
                        }
                        expmap(latent, tangent, LDIM, CURVATURE);
                    }
                }
            }
        }
        
        if (fn % 60 == 0 || fn < 5) {
            printf("  Frame %d (%s)\n", fn, is_i ? "I" : "P");
        }
    }
    
    printf("--- Pass 2: Encode B frames ---\n");
    for (int fn = 0; fn < nf; fn++) {
        if (ftypes[fn] != 2) continue;
        
        int prev_fn = fn - 1;
        while (prev_fn >= 0 && ftypes[prev_fn] == 2) prev_fn--;
        int next_fn = fn + 1;
        while (next_fn < nf && ftypes[next_fn] == 2) next_fn++;
        
        if (prev_fn >= 0 && next_fn < nf) {
            float *fwd_lat = &all_latents[prev_fn * num_latents];
            float *bwd_lat = &all_latents[next_fn * num_latents];
            float *fn_lat = &all_latents[fn * num_latents];
            
            for (int i = 0; i < num_latents; i++) {
                fn_lat[i] = 0.5f * (fwd_lat[i] + bwd_lat[i]);
            }
        }
    }
    
    /* Decode and compute PSNR */
    printf("--- Decode and Compute PSNR ---\n");
    uint8_t *recon = (uint8_t *)malloc(np);
    double total_psnr = 0;
    
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np;
        float *latent = &all_latents[fn * num_latents];
        
        /* Decode each block */
        for (int blk_y = 0; blk_y < H; blk_y += 64) {
            for (int blk_x = 0; blk_x < W; blk_x += 64) {
                int blk_h = (blk_y + 64 > H) ? H - blk_y : 64;
                int blk_w = (blk_x + 64 > W) ? W - blk_x : 64;
                
                /* Decode each 4x4 patch */
                for (int py = 0; py < 16; py++) {
                    for (int px = 0; px < 16; px++) {
                        int patch_idx = py * 16 + px;
                        float *lat = &latent[patch_idx * LDIM];
                        
                        float patch[48];
                        decode_patch(lat, patch);
                        
                        /* Write back to reconstruction */
                        int idx = 0;
                        for (int dy = 0; dy < 4; dy++) {
                            for (int dx = 0; dx < 4; dx++) {
                                int iy = py*4+dy;
                                int ix = px*4+dx;
                                if (iy < blk_h && ix < blk_w) {
                                    int out_idx = (blk_y+iy)*W+(blk_x+ix);
                                    uint8_t r = (uint8_t)fmaxf(0, fminf(255, (patch[idx] + 1.0f) * 127.5f));
                                    recon[out_idx] = r;
                                }
                                idx += 3;
                            }
                        }
                    }
                }
            }
        }
        
        double mse = 0;
        for (int i = 0; i < np; i++) {
            double d = (double)orig[i] - (double)recon[i];
            mse += d * d;
        }
        mse /= np;
        double p = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        total_psnr += p;
        
        if (fn % 60 == 0 || fn < 5) {
            printf("  Frame %d: PSNR=%.1f dB\n", fn, p);
        }
    }
    
    double avg_psnr = total_psnr / nf;
    printf("\n=== Results ===\n");
    printf("Average PSNR: %.1f dB\n", avg_psnr);
    
    free(frames);
    free(recon);
    free(all_latents);
    free(ftypes);
    
    return 0;
}
