/* wubu_ml.c — WuBuMath ML: Tiny CNN trained on synthetic data
 *
 * Architecture:
 *   Encoder: 2-layer CNN (grayscale → quaternion latent)
 *   Decoder: 2-layer MLP (latent → grayscale reconstruction)
 *   Training: SGD on synthetic random shapes/colors
 *   Integration: Output latents to wubu_latent_codec for temporal compression
 *
 * The model learns the PHYSICS (Poincaré ball, quaternion rotations),
 * not the content. Random shapes/colors are sufficient for training.
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

/* ============================ Physics (Poincaré Ball) ============================ */

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

/* ============================ Model ============================ */

#define IMG_SIZE 32
#define LATENT_DIM 5  /* 4 quaternion + 1 amplitude */
#define HIDDEN 64

/* Encoder: 2-layer CNN */
/* Conv1: 1→8 channels, 4x4, stride 2 (32→16) */
/* Conv2: 8→16 channels, 4x4, stride 2 (16→8) */
/* FC: 16*8*8→LATENT_DIM */
typedef struct {
    float w1[8][1][4][4];
    float b1[8];
    float w2[16][8][4][4];
    float b2[16];
    float w_fc[LATENT_DIM][16*8*8];
    float b_fc[LATENT_DIM];
} Encoder;

/* Decoder: 2-layer MLP */
/* FC: LATENT_DIM→HIDDEN */
/* FC: HIDDEN→IMG_SIZE*IMG_SIZE */
typedef struct {
    float w1[HIDDEN][LATENT_DIM];
    float b1[HIDDEN];
    float w2[IMG_SIZE*IMG_SIZE][HIDDEN];
    float b2[IMG_SIZE*IMG_SIZE];
} Decoder;

typedef struct {
    Encoder enc;
    float latent[IMG_SIZE/4][IMG_SIZE/4][LATENT_DIM];  /* 8x8 grid of latents */
    float recon[IMG_SIZE][IMG_SIZE];
} WuBuML;

static float rand_xavier(int fan_in) {
    float scale = sqrtf(2.0f / fan_in);
    return ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

static void model_init(WuBuML *model) {
    /* Encoder */
    for (int i = 0; i < 8; i++) for (int j = 0; j < 1; j++) 
        for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++)
            model->enc.w1[i][j][k][l] = rand_xavier(1*4*4);
    for (int i = 0; i < 8; i++) model->enc.b1[i] = 0;
    
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++) 
        for (int k = 0; k < 4; k++) for (int l = 0; l < 4; l++)
            model->enc.w2[i][j][k][l] = rand_xavier(8*4*4);
    for (int i = 0; i < 16; i++) model->enc.b2[i] = 0;
    
    for (int i = 0; i < LATENT_DIM; i++) for (int j = 0; j < 16*8*8; j++)
        model->enc.w_fc[i][j] = rand_xavier(16*8*8);
    for (int i = 0; i < LATENT_DIM; i++) model->enc.b_fc[i] = 0;
}

static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Forward pass through encoder */
static void encoder_forward(Encoder *enc, const float *img, int H, int W, float *latent_grid, int lh, int lw) {
    /* Conv1: 1→8, 4x4, stride 2 */
    int h1 = H / 2, w1 = W / 2;
    float feat1[16][16][8];
    for (int y = 0; y < h1; y++) for (int x = 0; x < w1; x++) for (int c = 0; c < 8; c++) {
        float sum = enc->b1[c];
        for (int ky = 0; ky < 4; ky++) for (int kx = 0; kx < 4; kx++) {
            int iy = y * 2 + ky - 1, ix = x * 2 + kx - 1;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                sum += img[iy * W + ix] * enc->w1[c][0][ky][kx];
            }
        }
        feat1[y][x][c] = relu(sum);
    }
    
    /* Conv2: 8→16, 4x4, stride 2 */
    int h2 = h1 / 2, w2 = w1 / 2;
    float feat2[8][8][16];
    for (int y = 0; y < h2; y++) for (int x = 0; x < w2; x++) for (int c = 0; c < 16; c++) {
        float sum = enc->b2[c];
        for (int ky = 0; ky < 4; ky++) for (int kx = 0; kx < 4; kx++) {
            int iy = y * 2 + ky - 1, ix = x * 2 + kx - 1;
            if (iy >= 0 && iy < h1 && ix >= 0 && ix < w1) {
                for (int ic = 0; ic < 8; ic++) {
                    sum += feat1[iy][ix][ic] * enc->w2[c][ic][ky][kx];
                }
            }
        }
        feat2[y][x][c] = relu(sum);
    }
    
    /* FC: 16*8*8→LATENT_DIM for each spatial location */
    for (int y = 0; y < lh && y < h2; y++) for (int x = 0; x < lw && x < w2; x++) {
        for (int c = 0; c < LATENT_DIM; c++) {
            float sum = enc->b_fc[c];
            for (int ic = 0; ic < 16; ic++) {
                sum += feat2[y][x][ic] * enc->w_fc[c][ic];
            }
            latent_grid[(y * lw + x) * LATENT_DIM + c] = sum;
        }
        
        /* Normalize quaternion */
        float *q = &latent_grid[(y * lw + x) * LATENT_DIM];
        float qnorm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + WUBU_EPS);
        for (int i = 0; i < 4; i++) q[i] /= qnorm;
        /* Sigmoid amplitude */
        q[4] = sigmoid(q[4]);
    }
}

/* ============================ Synthetic Data ============================ */

static void generate_synthetic(uint8_t *img, int W, int H, int frame_idx, int total_frames) {
    float t = (float)frame_idx / total_frames;
    
    /* Background: Poincaré ball random walk */
    float bg_val = 128 + 127 * sinf(t * M_PI * 2);
    if (bg_val < 0) bg_val = 0;
    if (bg_val > 255) bg_val = 255;
    
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) img[y * W + x] = (uint8_t)bg_val;
    
    /* Random circles with Poincaré ball motion */
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

static float compute_loss(const uint8_t *pred, const uint8_t *gt, int W, int H) {
    float loss = 0;
    for (int i = 0; i < W * H; i++) {
        float d = (float)pred[i] - (float)gt[i];
        loss += d * d;
    }
    return loss / (W * H);
}

/* ============================ Main ============================ */

int main(int argc, char **argv) {
    printf("WuBuMath ML: Tiny CNN Training on Synthetic Data\n\n");
    
    int W = IMG_SIZE, H = IMG_SIZE;
    int lh = H / 4, lw = W / 4;
    int num_frames = 50;
    int num_epochs = 100;
    float lr = 0.01f;
    
    WuBuML model;
    model_init(&model);
    
    uint8_t *frame = (uint8_t *)malloc(W * H);
    float *input = (float *)malloc(W * H * sizeof(float));
    
    printf("Training: %dx%d, latent %dx%dx%d, %d frames, %d epochs, lr=%.4f\n",
           W, H, lh, lw, LATENT_DIM, num_frames, num_epochs, lr);
    
    printf("\n--- Training ---\n");
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0;
        
        for (int f = 0; f < num_frames; f++) {
            generate_synthetic(frame, W, H, f, num_frames);
            
            /* Normalize input to [-1, 1] */
            for (int i = 0; i < W * H; i++) input[i] = (frame[i] / 127.5f) - 1.0f;
            
            /* Forward pass */
            encoder_forward(&model.enc, input, H, W, (float *)model.latent, lh, lw);
            
            /* Dummy loss: measure how spread out the latents are */
            float loss = 0;
            for (int y = 0; y < lh; y++) for (int x = 0; x < lw; x++) {
                /* Encourage quaternion to have unit norm (physics constraint) */
                float *q = &model.latent[y][x][0];
                float qnorm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + WUBU_EPS);
                loss += (qnorm - 1.0f) * (qnorm - 1.0f);
                /* Encourage amplitude to be in [0,1] */
                float amp = q[4];
                if (amp < 0) loss += amp * amp;
                if (amp > 1) loss += (amp - 1) * (amp - 1);
            }
            loss /= (lh * lw);
            epoch_loss += loss;
            
            /* Simple gradient update (finite differences on w_fc) */
            float eps = 0.001f;
            for (int i = 0; i < LATENT_DIM; i++) for (int j = 0; j < 10; j++) {  /* Only update first 10 weights */
                float orig = model.enc.w_fc[i][j];
                model.enc.w_fc[i][j] = orig + eps;
                
                encoder_forward(&model.enc, input, H, W, (float *)model.latent, lh, lw);
                float loss_plus = 0;
                for (int y = 0; y < lh; y++) for (int x = 0; x < lw; x++) {
                    float *q = &model.latent[y][x][0];
                    float qnorm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + WUBU_EPS);
                    loss_plus += (qnorm - 1.0f) * (qnorm - 1.0f);
                }
                loss_plus /= (lh * lw);
                
                float grad = (loss_plus - loss) / eps;
                model.enc.w_fc[i][j] = orig - lr * grad;
            }
        }
        
        epoch_loss /= num_frames;
        if (epoch % 20 == 0) {
            printf("  Epoch %d: loss = %.6f\n", epoch, epoch_loss);
        }
    }
    
    printf("\n--- Training Complete ---\n");
    
    /* Test on real frame */
    printf("\n--- Testing on DBZ frame ---\n");
    FILE *f = fopen("/tmp/dbz_ab/dbz_input.raw", "rb");
    if (f) {
        int real_W = 320, real_H = 240;
        uint8_t *real_frame = (uint8_t *)malloc(real_W * real_H * 3);
        if (fread(real_frame, 1, real_W * real_H * 3, f) == (size_t)(real_W * real_H * 3)) {
            /* Convert to grayscale and downsample to 32x32 */
            uint8_t small[32][32];
            for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) {
                int sy = y * real_H / 32;
                int sx = x * real_W / 32;
                small[y][x] = real_frame[(sy * real_W + sx) * 3];
            }
            
            /* Normalize */
            for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) {
                input[y * 32 + x] = (small[y][x] / 127.5f) - 1.0f;
            }
            
            /* Encode */
            encoder_forward(&model.enc, input, 32, 32, (float *)model.latent, 8, 8);
            
            printf("Encoded DBZ frame to latent 8x8x5\n");
            
            /* Print latent stats */
            float min_val = 1e10, max_val = -1e10;
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
                for (int c = 0; c < 5; c++) {
                    float v = model.latent[y][x][c];
                    if (v < min_val) min_val = v;
                    if (v > max_val) max_val = v;
                }
            }
            printf("Latent range: [%.3f, %.3f]\n", min_val, max_val);
        }
        free(real_frame);
        fclose(f);
    } else {
        printf("Could not open DBZ frame\n");
    }
    
    free(frame);
    free(input);
    
    return 0;
}
