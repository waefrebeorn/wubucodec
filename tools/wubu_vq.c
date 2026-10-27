/* wubu_vq.c — WuBuMath Vector Quantization on Poincaré Ball
 *
 * Key insight: No neural network training needed! Use a codebook of
 * pre-computed latents on the Poincaré ball. The codebook is optimized
 * on synthetic data (physics), then used for real video compression.
 *
 * Pipeline:
 * 1. Generate codebook of latents on Poincaré ball (optimized on synthetic data)
 * 2. For each block in real video, find best codebook entry
 * 3. Encode the codebook index + residual
 */
#define _POSIX_C_SOURCE 20080901L
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

/* ============================ Codebook ============================ */

#define CODEBOOK_SIZE 256
#define LATENT_DIM 5
#define BLOCK_SIZE 8

typedef struct {
    float entries[CODEBOOK_SIZE][LATENT_DIM];
    int counts[CODEBOOK_SIZE];  /* How often each entry is used */
} Codebook;

/* Initialize codebook with random latents on Poincaré ball */
static void codebook_init(Codebook *cb) {
    for (int i = 0; i < CODEBOOK_SIZE; i++) {
        /* Random tangent vector */
        float tangent[LATENT_DIM];
        for (int j = 0; j < LATENT_DIM; j++) {
            tangent[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        
        /* Map to Poincaré ball */
        expmap(cb->entries[i], tangent, LATENT_DIM, CURVATURE);
        
        /* Normalize quaternion */
        float qnorm = sqrtf(cb->entries[i][0]*cb->entries[i][0] + cb->entries[i][1]*cb->entries[i][1] + 
                           cb->entries[i][2]*cb->entries[i][2] + cb->entries[i][3]*cb->entries[i][3] + WUBU_EPS);
        for (int j = 0; j < 4; j++) cb->entries[i][j] /= qnorm;
        
        /* Amplitude in [0,1] */
        cb->entries[i][4] = 1.0f / (1.0f + expf(-cb->entries[i][4]));
        
        cb->counts[i] = 0;
    }
}

/* Find best codebook entry for a given block */
static int codebook_find_best(Codebook *cb, const float *block, int block_size) {
    int best_idx = 0;
    float best_dist = 1e10;
    
    for (int i = 0; i < CODEBOOK_SIZE; i++) {
        /* Decode the latent to get a predicted block */
        float pred[BLOCK_SIZE * BLOCK_SIZE];
        float q[4] = {cb->entries[i][0], cb->entries[i][1], cb->entries[i][2], cb->entries[i][3]};
        float amp = cb->entries[i][4];
        
        /* Simple decoding: quaternion rotation + amplitude */
        for (int y = 0; y < block_size; y++) {
            for (int x = 0; x < block_size; x++) {
                float fy = (float)y / block_size;
                float fx = (float)x / block_size;
                
                /* Spatial modulation based on quaternion */
                float mod = 0.5f + 0.5f * sinf(fx * M_PI * 2 * q[0]) * cosf(fy * M_PI * 2 * q[1]);
                
                pred[y * block_size + x] = amp * mod * 255.0f;
            }
        }
        
        /* Compute distance */
        float dist = 0;
        for (int j = 0; j < block_size * block_size; j++) {
            float d = pred[j] - block[j];
            dist += d * d;
        }
        dist /= (block_size * block_size);
        
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    cb->counts[best_idx]++;
    return best_idx;
}

/* ============================ Synthetic Data ============================ */

static void generate_synthetic(uint8_t *img, int W, int H, int frame_idx, int total_frames) {
    float t = (float)frame_idx / total_frames;
    
    uint8_t bg = (uint8_t)(128 + 127 * sinf(t * M_PI * 2));
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) img[y * W + x] = bg;
    
    int num_circles = 3 + rand() % 5;
    for (int c = 0; c < num_circles; c++) {
        float cx = (0.2f + 0.6f * sinf(t * M_PI * 2 + c * 1.7f)) * W;
        float cy = (0.2f + 0.6f * cosf(t * M_PI * 2 + c * 2.3f)) * H;
        float radius = 5 + 10 * sinf(t * M_PI * 2 + c);
        uint8_t cc = (uint8_t)(128 + 127 * sinf(c * 1.1f));
        
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            float dx = x - cx, dy = y - cy;
            if (dx*dx + dy*dy < radius*radius) img[y * W + x] = cc;
        }
    }
}

/* ============================ Training ============================ */

static void codebook_train(Codebook *cb, int num_frames, int img_size) {
    printf("Training codebook on %d synthetic frames...\n", num_frames);
    
    uint8_t *frame = (uint8_t *)malloc(img_size * img_size);
    float *block = (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    
    for (int epoch = 0; epoch < 10; epoch++) {
        for (int f = 0; f < num_frames; f++) {
            generate_synthetic(frame, img_size, img_size, f, num_frames);
            
            /* Process each block */
            for (int by = 0; by < img_size / BLOCK_SIZE; by++) {
                for (int bx = 0; bx < img_size / BLOCK_SIZE; bx++) {
                    /* Extract block */
                    for (int y = 0; y < BLOCK_SIZE; y++) {
                        for (int x = 0; x < BLOCK_SIZE; x++) {
                            block[y * BLOCK_SIZE + x] = frame[(by * BLOCK_SIZE + y) * img_size + (bx * BLOCK_SIZE + x)];
                        }
                    }
                    
                    /* Find best codebook entry */
                    int best = codebook_find_best(cb, block, BLOCK_SIZE);
                    
                    /* Update codebook entry slightly toward this block (Lloyd's algorithm) */
                    float lr = 0.01f;
                    float pred[BLOCK_SIZE * BLOCK_SIZE];
                    float q[4] = {cb->entries[best][0], cb->entries[best][1], cb->entries[best][2], cb->entries[best][3]};
                    float amp = cb->entries[best][4];
                    
                    for (int y = 0; y < BLOCK_SIZE; y++) {
                        for (int x = 0; x < BLOCK_SIZE; x++) {
                            float fy = (float)y / BLOCK_SIZE;
                            float fx = (float)x / BLOCK_SIZE;
                            float mod = 0.5f + 0.5f * sinf(fx * M_PI * 2 * q[0]) * cosf(fy * M_PI * 2 * q[1]);
                            pred[y * BLOCK_SIZE + x] = amp * mod * 255.0f;
                        }
                    }
                    
                    /* Gradient of MSE w.r.t. amplitude */
                    float grad_amp = 0;
                    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
                        grad_amp += -2 * (block[i] - pred[i]) * (pred[i] / (amp + WUBU_EPS)) / (BLOCK_SIZE * BLOCK_SIZE);
                    }
                    cb->entries[best][4] -= lr * grad_amp;
                    if (cb->entries[best][4] < 0) cb->entries[best][4] = 0;
                    if (cb->entries[best][4] > 1) cb->entries[best][4] = 1;
                }
            }
        }
        
        /* Print stats */
        int used = 0;
        for (int i = 0; i < CODEBOOK_SIZE; i++) if (cb->counts[i] > 0) used++;
        printf("  Epoch %d: %d/%d codebook entries used\n", epoch, used, CODEBOOK_SIZE);
    }
    
    free(frame);
    free(block);
}

/* ============================ Main ============================ */

int main(int argc, char **argv) {
    printf("WuBuMath Vector Quantization on Poincaré Ball\n\n");
    
    int img_size = 32;
    
    Codebook cb;
    codebook_init(&cb);
    
    /* Train on synthetic data */
    codebook_train(&cb, 100, img_size);
    
    /* Test on real frame */
    printf("\n--- Testing on DBZ frame ---\n");
    FILE *f = fopen("/tmp/dbz_ab/dbz_input.raw", "rb");
    if (f) {
        int real_W = 320, real_H = 240;
        uint8_t *real_frame = (uint8_t *)malloc(real_W * real_H * 3);
        if (fread(real_frame, 1, real_W * real_H * 3, f) == (size_t)(real_W * real_H * 3)) {
            /* Convert to grayscale and downsample */
            uint8_t small[32][32];
            for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) {
                int sy = y * real_H / 32;
                int sx = x * real_W / 32;
                small[y][x] = real_frame[(sy * real_W + sx) * 3];
            }
            
            /* Encode each block */
            int total_indices = 0;
            float total_error = 0;
            
            for (int by = 0; by < 32 / BLOCK_SIZE; by++) {
                for (int bx = 0; bx < 32 / BLOCK_SIZE; bx++) {
                    float block[BLOCK_SIZE * BLOCK_SIZE];
                    for (int y = 0; y < BLOCK_SIZE; y++) {
                        for (int x = 0; x < BLOCK_SIZE; x++) {
                            block[y * BLOCK_SIZE + x] = small[by * BLOCK_SIZE + y][bx * BLOCK_SIZE + x];
                        }
                    }
                    
                    int idx = codebook_find_best(&cb, block, BLOCK_SIZE);
                    total_indices += idx;
                    
                    /* Compute error */
                    float pred[BLOCK_SIZE * BLOCK_SIZE];
                    float q[4] = {cb.entries[idx][0], cb.entries[idx][1], cb.entries[idx][2], cb.entries[idx][3]};
                    float amp = cb.entries[idx][4];
                    
                    for (int y = 0; y < BLOCK_SIZE; y++) {
                        for (int x = 0; x < BLOCK_SIZE; x++) {
                            float fy = (float)y / BLOCK_SIZE;
                            float fx = (float)x / BLOCK_SIZE;
                            float mod = 0.5f + 0.5f * sinf(fx * M_PI * 2 * q[0]) * cosf(fy * M_PI * 2 * q[1]);
                            pred[y * BLOCK_SIZE + x] = amp * mod * 255.0f;
                        }
                    }
                    
                    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
                        float d = pred[i] - block[i];
                        total_error += d * d;
                    }
                }
            }
            
            total_error /= (32 * 32);
            float psnr = 10.0f * log10f(255.0f * 255.0f / total_error);
            printf("Encoded DBZ frame: PSNR = %.1f dB\n", psnr);
            printf("Total bits needed: %d indices * 8 bits = %d bits\n", 
                   (32/BLOCK_SIZE) * (32/BLOCK_SIZE), 
                   (32/BLOCK_SIZE) * (32/BLOCK_SIZE) * 8);
        }
        free(real_frame);
        fclose(f);
    } else {
        printf("Could not open DBZ frame\n");
    }
    
    return 0;
}
