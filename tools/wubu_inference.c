/* wubu_inference.c — WuBuMath Inference with Trained Model
 *
 * Uses the ACTUAL trained model from train_wubu_codec.c.
 * Implements the full forward pass:
 *   Encoder: patch[48] → Linear[48→32] → LayerNorm → 4 flow MLP levels → latent[32]
 *   Decoder: latent[32] → Linear[32→48] → reconstructed patch[48]
 *
 * The model was trained on synthetic data and achieved 31.9 dB.
 * This tests if it generalizes to real video.
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

#define WUBU_EPS 1e-6f

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
    size_t rd = fread(&model, sizeof(Model), 1, f);
    fclose(f);
    printf("Loaded model from %s (%zu bytes)\n", path, rd * sizeof(Model));
}

/* ============================ Forward Pass ============================ */

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

/* Encode one patch: patch[48] → latent[32] */
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
    for (int i = 0; i < DIM; i++) { float d = pre[i] - mean; var += d*d; }
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

/* Decode one patch: latent[32] → patch[48] */
static void decode_patch(const float *latent, float *patch) {
    for (int o = 0; o < PP; o++) {
        float pred = model.db[o];
        for (int i = 0; i < LDIM; i++) pred += model.dw[o*LDIM+i] * latent[i];
        patch[o] = pred;
    }
}

/* ============================ Image Processing ============================ */

/* Encode full image: 64x64 RGB → 256 latents[32] */
static void encode_image(const float *img, float *latents) {
    for (int p = 0; p < NP; p++) {
        int py = p / NPW, px = p % NPW;
        
        /* Extract patch pixels */
        float patch[PP];
        int idx = 0;
        for (int dy = 0; dy < PS; dy++) {
            for (int dx = 0; dx < PS; dx++) {
                int iy = py * PS + dy, ix = px * PS + dx;
                int ii = (iy * IMG_W + ix) * 3;
                patch[idx++] = img[ii];
                patch[idx++] = img[ii+1];
                patch[idx++] = img[ii+2];
            }
        }
        
        /* Encode */
        encode_patch(patch, &latents[p * LDIM]);
    }
}

/* Decode full image: 256 latents[32] → 64x64 RGB */
static void decode_image(const float *latents, float *img) {
    for (int p = 0; p < NP; p++) {
        int py = p / NPW, px = p % NPW;
        
        /* Decode */
        float patch[PP];
        decode_patch(&latents[p * LDIM], patch);
        
        /* Write back */
        int idx = 0;
        for (int dy = 0; dy < PS; dy++) {
            for (int dx = 0; dx < PS; dx++) {
                int iy = py * PS + dy, ix = px * PS + dx;
                int ii = (iy * IMG_W + ix) * 3;
                img[ii]     = patch[idx++];
                img[ii + 1] = patch[idx++];
                img[ii + 2] = patch[idx++];
            }
        }
    }
}

/* ============================ Main ============================ */

int main(int argc, char **argv) {
    if (argc < 2) { printf("Usage: %s input.raw [options]\n", argv[0]); return 1; }

    const char *input = argv[1];
    int W = 320, H = 240;
    
    for (int i = 2; i < argc; i += 2) {
        if (i+1 >= argc) break;
        if (strcmp(argv[i], "-w") == 0) W = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-h") == 0) H = atoi(argv[i+1]);
    }
    
    int np = W * H;
    
    /* Load trained model */
    load_model("/home/wubu/WuBuMath/output/wubu_trained_final.bin");
    
    /* Load input frame */
    FILE *f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    int nf = (int)(fsz / np);
    uint8_t *frames = (uint8_t *)malloc((size_t)nf * np);
    if (fread(frames, 1, (size_t)nf*np, f) != (size_t)(nf*np)) { printf("Read failed\n"); return 1; }
    fclose(f);
    printf("Loaded %d frames (%dx%d)\n", nf, W, H);
    
    /* Process first frame as test */
    const uint8_t *orig = frames;
    
    /* Downsample to 64x64 and convert to float [0,1] */
    float small[64*64*3];
    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) {
        int sy = y * H / 64;
        int sx = x * W / 64;
        small[(y*64+x)*3+0] = orig[(sy*W+sx)*3+0] / 255.0f;
        small[(y*64+x)*3+1] = orig[(sy*W+sx)*3+1] / 255.0f;
        small[(y*64+x)*3+2] = orig[(sy*W+sx)*3+2] / 255.0f;
    }
    
    /* Encode to latents */
    float latents[NP * LDIM];
    encode_image(small, latents);
    
    printf("Encoded to %d latents[%d]\n", NP, LDIM);
    
    /* Decode */
    float decoded[64*64*3];
    decode_image(latents, decoded);
    
    /* Compute PSNR */
    double mse = 0;
    for (int i = 0; i < 64*64*3; i++) {
        double d = decoded[i] - small[i];
        mse += d * d;
    }
    mse /= (64*64*3);
    double psnr = (mse > 0) ? 10.0 * log10(1.0 / mse) : 99.0;
    printf("Reconstruction PSNR: %.1f dB\n", psnr);
    
    /* Print some stats */
    float min_val = 1e10, max_val = -1e10;
    for (int i = 0; i < NP * LDIM; i++) {
        if (latents[i] < min_val) min_val = latents[i];
        if (latents[i] > max_val) max_val = latents[i];
    }
    printf("Latent range: [%.3f, %.3f]\n", min_val, max_val);
    
    free(frames);
    return 0;
}
