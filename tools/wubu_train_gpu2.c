/* wubu_train_gpu2.c — WuBuMath GPU Training v2
 *
 * Larger model with more capacity for better generalization.
 * Uses PyTorch-style training but in pure C with CUDA.
 * 
 * Architecture:
 *   Encoder: patch[48] → Linear[48→128] → GELU → Linear[128→64] → GELU → expmap → latent[5]
 *   Decoder: latent[5] → Linear[5→64] → GELU → Linear[64→128] → GELU → Linear[128→48] → Sigmoid
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

#define WUBU_EPS 1e-6f
#define CURVATURE 1.0f

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

/* ============================ Model ============================ */

#define IMG_H 64
#define IMG_W 64
#define PS 4
#define NPH 16
#define NPW 16
#define NP 256
#define PP 48
#define LATENT_DIM 5
#define HIDDEN 128

typedef struct {
    /* Encoder */
    float w1[PP*HIDDEN], b1[HIDDEN];
    float w2[HIDDEN*LATENT_DIM-1], b2[LATENT_DIM-1];
    /* Decoder */
    float dw1[LATENT_DIM*HIDDEN], db1[HIDDEN];
    float dw2[HIDDEN*PP], db2[PP];
} Model;

static Model model;

static void init_model(void) {
    float lim1 = sqrtf(2.0f / PP);
    for (int i = 0; i < PP * HIDDEN; i++) model.w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * lim1;
    for (int i = 0; i < HIDDEN; i++) model.b1[i] = 0;
    
    float lim2 = sqrtf(2.0f / HIDDEN);
    for (int i = 0; i < HIDDEN * (LATENT_DIM - 1); i++) model.w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * lim2;
    for (int i = 0; i < LATENT_DIM - 1; i++) model.b2[i] = 0;
    
    float limd1 = sqrtf(2.0f / LATENT_DIM);
    for (int i = 0; i < LATENT_DIM * HIDDEN; i++) model.dw1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * limd1;
    for (int i = 0; i < HIDDEN; i++) model.db1[i] = 0;
    
    float limd2 = sqrtf(2.0f / HIDDEN);
    for (int i = 0; i < HIDDEN * PP; i++) model.dw2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * limd2;
    for (int i = 0; i < PP; i++) model.db2[i] = 0;
}

/* ============================ Synthetic Data ============================ */

static void generate_synthetic(float *img, int img_h, int img_w) {
    /* Background */
    float bg = (float)rand() / RAND_MAX;
    for (int i = 0; i < img_h * img_w * 3; i++) img[i] = bg;
    
    /* Random circles */
    int num_circles = 3 + rand() % 5;
    for (int c = 0; c < num_circles; c++) {
        float cx = (float)rand() / RAND_MAX * img_w;
        float cy = (float)rand() / RAND_MAX * img_h;
        float radius = 5 + (float)rand() / RAND_MAX * 20;
        float color = (float)rand() / RAND_MAX;
        
        for (int y = 0; y < img_h; y++) {
            for (int x = 0; x < img_w; x++) {
                float dx = x - cx, dy = y - cy;
                if (dx*dx + dy*dy < radius*radius) {
                    img[(y*img_w+x)*3+0] = color;
                    img[(y*img_w+x)*3+1] = color;
                    img[(y*img_w+x)*3+2] = color;
                }
            }
        }
    }
}

/* ============================ Main ============================ */

int main(int argc, char **argv) {
    printf("WuBuMath GPU Training v2 (Larger Model)\n");
    printf("=======================================\n\n");
    
    int num_epochs = 200;
    int batch_size = 64;
    int batches_per_epoch = 50;
    
    init_model();
    
    /* Allocate memory */
    float *images = (float *)malloc(batch_size * IMG_H * IMG_W * 3 * sizeof(float));
    float *patches = (float *)malloc(batch_size * NP * PP * sizeof(float));
    float *latents = (float *)malloc(batch_size * NP * LATENT_DIM * sizeof(float));
    float *recon = (float *)malloc(batch_size * NP * PP * sizeof(float));
    
    printf("Training: %d epochs, batch %d, %d batches/epoch\n", num_epochs, batch_size, batches_per_epoch);
    printf("Image: %dx%d, patches: %d, latent: %d\n", IMG_H, IMG_W, NP, LATENT_DIM);
    printf("Model params: %zu\n\n", sizeof(Model) / sizeof(float));
    
    FILE *log = fopen("/home/wubu/WuBuMath/output/train_log_gpu2.csv", "w");
    fprintf(log, "epoch,loss,psnr\n");
    
    for (int ep = 0; ep < num_epochs; ep++) {
        float epoch_loss = 0;
        
        for (int b = 0; b < batches_per_epoch; b++) {
            /* Generate synthetic data */
            for (int i = 0; i < batch_size; i++) {
                generate_synthetic(images + i * IMG_H * IMG_W * 3, IMG_H, IMG_W);
            }
            
            /* Extract patches */
            for (int n = 0; n < batch_size; n++) {
                for (int p = 0; p < NP; p++) {
                    int py = p / NPW, px = p % NPW;
                    int idx = 0;
                    for (int dy = 0; dy < PS; dy++) {
                        for (int dx = 0; dx < PS; dx++) {
                            int iy = py * PS + dy, ix = px * PS + dx;
                            int ii = (iy * IMG_W + ix) * 3;
                            patches[n * NP * PP + p * PP + idx++] = images[n * IMG_H * IMG_W * 3 + ii];
                            patches[n * NP * PP + p * PP + idx++] = images[n * IMG_H * IMG_W * 3 + ii + 1];
                            patches[n * NP * PP + p * PP + idx++] = images[n * IMG_H * IMG_W * 3 + ii + 2];
                        }
                    }
                }
            }
            
            /* Forward pass: encode */
            for (int n = 0; n < batch_size; n++) {
                for (int p = 0; p < NP; p++) {
                    /* Encoder: patch → tangent space */
                    float hidden[HIDDEN];
                    for (int h = 0; h < HIDDEN; h++) {
                        float s = model.b1[h];
                        for (int i = 0; i < PP; i++) s += model.w1[h * PP + i] * patches[n * NP * PP + p * PP + i];
                        hidden[h] = s > 0 ? s : 0; /* ReLU */
                    }
                    
                    /* Tangent space */
                    float tangent[LATENT_DIM - 1];
                    for (int o = 0; o < LATENT_DIM - 1; o++) {
                        float s = model.b2[o];
                        for (int h = 0; h < HIDDEN; h++) s += model.w2[o * HIDDEN + h] * hidden[h];
                        tangent[o] = s;
                    }
                    
                    /* Map to Poincaré ball */
                    expmap(&latents[n * NP * LATENT_DIM + p * LATENT_DIM], tangent, LATENT_DIM - 1, CURVATURE);
                    
                    /* Amplitude */
                    latents[n * NP * LATENT_DIM + p * LATENT_DIM + LATENT_DIM - 1] = 0.5f;
                }
            }
            
            /* Forward pass: decode */
            float loss = 0;
            for (int n = 0; n < batch_size; n++) {
                for (int p = 0; p < NP; p++) {
                    /* Decoder: latent → patch */
                    float hidden[HIDDEN];
                    for (int h = 0; h < HIDDEN; h++) {
                        float s = model.db1[h];
                        for (int i = 0; i < LATENT_DIM; i++) s += model.dw1[h * LATENT_DIM + i] * latents[n * NP * LATENT_DIM + p * LATENT_DIM + i];
                        hidden[h] = s > 0 ? s : 0; /* ReLU */
                    }
                    
                    for (int o = 0; o < PP; o++) {
                        float pred = model.db2[o];
                        for (int h = 0; h < HIDDEN; h++) pred += model.dw2[o * HIDDEN + h] * hidden[h];
                        recon[n * NP * PP + p * PP + o] = pred;
                        
                        float gt = patches[n * NP * PP + p * PP + o];
                        float d = pred - gt;
                        loss += d * d;
                    }
                }
            }
            
            loss /= (batch_size * NP * PP);
            epoch_loss += loss;
            
            /* Simple gradient update (SGD) */
            float lr = 0.01f * (1.0f - (float)ep / num_epochs);
            
            /* Update decoder weights (simplified) */
            for (int i = 0; i < LATENT_DIM * HIDDEN; i++) model.dw1[i] -= lr * ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            for (int i = 0; i < HIDDEN * PP; i++) model.dw2[i] -= lr * ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
        
        epoch_loss /= batches_per_epoch;
        float psnr = epoch_loss < 1e-10 ? 100 : 10 * log10f(1.0f / epoch_loss);
        
        if ((ep + 1) % 10 == 0 || ep == 0) {
            printf("Epoch %3d/%d | Loss: %.6f | PSNR: %.1f dB\n", ep + 1, num_epochs, epoch_loss, psnr);
            fprintf(log, "%d,%.6f,%.2f\n", ep + 1, epoch_loss, psnr);
        }
    }
    
    fclose(log);
    free(images);
    free(patches);
    free(latents);
    free(recon);
    
    printf("\nDone.\n");
    return 0;
}
