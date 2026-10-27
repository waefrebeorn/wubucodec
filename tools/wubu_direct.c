/* wubu_direct.c — WuBuMath Direct Optimization
 *
 * Instead of backpropagation (too slow in C), use DIRECT OPTIMIZATION.
 * For each frame, find the Poincaré ball latent that minimizes reconstruction error.
 * This is like vector quantization on the manifold.
 *
 * The model learns the PHYSICS of the Poincaré ball, not the content.
 * Random shapes/colors are the training data.
 *
 * Pipeline:
 * 1. Generate synthetic frame
 * 2. Optimize latent to minimize reconstruction error (direct search on manifold)
 * 3. The optimization learns the mapping implicitly
 * 4. Use the optimized latent for compression
 */
#define _POSIX_C_SOURCE 20080911L
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

/* ============================ Direct Optimization ============================ */

#define IMG_SIZE 16  /* Small for fast optimization */
#define LATENT_DIM 5

/* Simple decoder: latent → image (fixed random projection + nonlinearity) */
static void decode_latent(const float *latent, float *img, int img_size) {
    /* Use the quaternion to rotate a base color, amplitude controls brightness */
    float q[4] = {latent[0], latent[1], latent[2], latent[3]};
    float amp = latent[4];
    
    /* Simple rotation: map quaternion to RGB color */
    float r = 0.5f + 0.5f * (q[0] + q[1]) * amp;
    float g = 0.5f + 0.5f * (q[2] + q[3]) * amp;
    float b = 0.5f + 0.5f * (q[0] - q[1]) * amp;
    
    /* Create image with spatial variation based on quaternion */
    for (int y = 0; y < img_size; y++) {
        for (int x = 0; x < img_size; x++) {
            float fy = (float)y / img_size;
            float fx = (float)x / img_size;
            
            /* Spatial modulation */
            float mod = 0.5f + 0.5f * sinf(fx * M_PI * 2 * q[0]) * cosf(fy * M_PI * 2 * q[1]);
            
            img[(y * img_size + x) * 3 + 0] = r * mod;
            img[(y * img_size + x) * 3 + 1] = g * mod;
            img[(y * img_size + x) * 3 + 2] = b * mod;
        }
    }
}

/* Compute reconstruction error */
static float compute_error(const float *pred, const uint8_t *target, int img_size) {
    float error = 0;
    for (int i = 0; i < img_size * img_size; i++) {
        float pr = pred[i * 3 + 0];
        float pg = pred[i * 3 + 1];
        float pb = pred[i * 3 + 2];
        
        float tr = target[i] / 255.0f;
        float tg = target[i] / 255.0f;
        float tb = target[i] / 255.0f;
        
        float dr = pr - tr;
        float dg = pg - tg;
        float db = pb - tb;
        
        error += dr * dr + dg * dg + db * db;
    }
    return error / (img_size * img_size);
}

/* Optimize latent to minimize reconstruction error */
static void optimize_latent(const uint8_t *target, float *latent, int img_size) {
    float best_error = 1e10;
    float best_latent[LATENT_DIM];
    
    /* Random search on Poincaré ball */
    for (int trial = 0; trial < 100; trial++) {
        /* Random tangent vector */
        float tangent[LATENT_DIM];
        for (int i = 0; i < LATENT_DIM; i++) {
            tangent[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        
        /* Map to Poincaré ball */
        float candidate[LATENT_DIM];
        expmap(candidate, tangent, LATENT_DIM, CURVATURE);
        
        /* Normalize quaternion */
        float qnorm = sqrtf(candidate[0]*candidate[0] + candidate[1]*candidate[1] + 
                           candidate[2]*candidate[2] + candidate[3]*candidate[3] + WUBU_EPS);
        for (int i = 0; i < 4; i++) candidate[i] /= qnorm;
        candidate[4] = 1.0f / (1.0f + expf(-candidate[4])); /* Sigmoid */
        
        /* Decode and compute error */
        float pred[IMG_SIZE * IMG_SIZE * 3];
        decode_latent(candidate, pred, img_size);
        float error = compute_error(pred, target, img_size);
        
        if (error < best_error) {
            best_error = error;
            memcpy(best_latent, candidate, sizeof(best_latent));
        }
    }
    
    memcpy(latent, best_latent, sizeof(best_latent));
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

/* ============================ Main ============================ */

int main(int argc, char **argv) {
    printf("WuBuMath Direct Optimization\n\n");
    
    int W = IMG_SIZE, H = IMG_SIZE;
    int num_frames = 100;
    
    uint8_t *frame = (uint8_t *)malloc(W * H);
    float *latent = (float *)malloc(LATENT_DIM * sizeof(float));
    
    printf("Training: %dx%d, %d frames\n", W, H, num_frames);
    
    printf("\n--- Direct Optimization ---\n");
    float total_error = 0;
    
    for (int f = 0; f < num_frames; f++) {
        generate_synthetic(frame, W, H, f, num_frames);
        
        /* Optimize latent to match this frame */
        optimize_latent(frame, latent, W);
        
        /* Decode and compute error */
        float pred[IMG_SIZE * IMG_SIZE * 3];
        decode_latent(latent, pred, W);
        float error = compute_error(pred, frame, W);
        total_error += error;
        
        if (f % 20 == 0) {
            printf("  Frame %d: error = %.4f, latent = [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
                   f, error, latent[0], latent[1], latent[2], latent[3], latent[4]);
        }
    }
    
    printf("\n--- Results ---\n");
    printf("Average error: %.4f\n", total_error / num_frames);
    
    /* Test on real frame */
    printf("\n--- Testing on DBZ frame ---\n");
    FILE *f = fopen("/tmp/dbz_ab/dbz_input.raw", "rb");
    if (f) {
        int real_W = 320, real_H = 240;
        uint8_t *real_frame = (uint8_t *)malloc(real_W * real_H * 3);
        if (fread(real_frame, 1, real_W * real_H * 3, f) == (size_t)(real_W * real_H * 3)) {
            /* Convert to grayscale and downsample */
            uint8_t small[IMG_SIZE][IMG_SIZE];
            for (int y = 0; y < IMG_SIZE; y++) for (int x = 0; x < IMG_SIZE; x++) {
                int sy = y * real_H / IMG_SIZE;
                int sx = x * real_W / IMG_SIZE;
                small[y][x] = real_frame[(sy * real_W + sx) * 3];
            }
            
            /* Optimize latent */
            optimize_latent((uint8_t *)small, latent, IMG_SIZE);
            
            printf("Optimized latent for DBZ frame: [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
                   latent[0], latent[1], latent[2], latent[3], latent[4]);
            
            /* Decode */
            float pred[IMG_SIZE * IMG_SIZE * 3];
            decode_latent(latent, pred, IMG_SIZE);
            float error = compute_error(pred, (uint8_t *)small, IMG_SIZE);
            printf("Reconstruction error: %.4f\n", error);
        }
        free(real_frame);
        fclose(f);
    } else {
        printf("Could not open DBZ frame\n");
    }
    
    free(frame);
    free(latent);
    
    return 0;
}
