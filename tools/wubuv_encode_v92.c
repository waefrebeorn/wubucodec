/* wubuv_encode_v92.c — WuBuCodec v92: Full SOTA pipeline */
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

#define DIM 32
#define LDIM 32
#define NUM_PATCH 256
#define PATCH_PIX 16
#define CONV_CH 32
#define TEMP_DIM 64

typedef struct {
    uint32_t low, range, value;
    int bits_pending;
    uint8_t *output;
    int output_pos, output_size;
} RangeEncoder;

typedef struct {
    uint16_t freq[258];
    uint16_t total, symbols, update_rate;
} ProbModel;

static void range_init(RangeEncoder *e, uint8_t *out, int sz) {
    e->low = 0; e->range = 0xFFFFFFFF; e->value = 0; e->bits_pending = 0;
    e->output = out; e->output_pos = 0; e->output_size = sz;
}

static void range_enc(RangeEncoder *e, ProbModel *m, uint32_t s) {
    uint32_t c = 0;
    for (uint32_t i = 0; i < s; i++) c += m->freq[i];
    uint32_t fs = m->freq[s];
    e->range /= m->total;
    e->low += c * e->range;
    e->range *= fs;
    while (e->range < 0x1000000) {
        e->range <<= 8;
        if (e->low < 0xFF000000 || (e->low >> 31)) {
            if (e->output_pos < e->output_size) e->output[e->output_pos++] = (uint8_t)((e->value >> 24) & 0xFF);
            e->value = (e->value << 8) & 0xFFFFFF00;
        }
        e->low = (e->low << 8) & 0xFFFFFFFF;
    }
    m->freq[s] += m->update_rate;
    m->total += m->update_rate;
    if (m->total > 16000) {
        m->total = 0;
        for (uint32_t i = 0; i <= m->symbols; i++) {
            m->freq[i] = (m->freq[i] + 1) >> 1;
            m->total += m->freq[i];
        }
    }
}

static int range_fin(RangeEncoder *e) {
    for (int i = 0; i < 4; i++) {
        if (e->output_pos < e->output_size) e->output[e->output_pos++] = (uint8_t)((e->low >> 24) & 0xFF);
        e->low <<= 8;
    }
    return e->output_pos;
}

static void model_init(ProbModel *m, uint32_t n, float d) {
    m->symbols = n; m->total = 0; m->update_rate = 4;
    for (uint32_t i = 0; i < n; i++) {
        m->freq[i] = (uint16_t)(powf(1.0f - d, (float)i) * 100.0f + 1.0f);
        m->total += m->freq[i];
    }
    m->freq[n] = 0;
}

typedef struct {
    float pw[1*DIM], pb[DIM];
    float dw[NUM_PATCH*PATCH_PIX*LDIM], db[NUM_PATCH*PATCH_PIX];
    struct { int dim, nb, fh; float c; float bd[4*32], desc[32], spr; float fw1[32*97], fb1[32], fw2[32*32], fb2[32]; } levels[4];
    float conv1_w[32*1*3*3], conv1_b[32];
    float conv2_w[DIM*32*3*3], conv2_b[DIM];
    float dconv1_w[32*DIM*3*3], dconv1_b[32];
    float dconv2_w[1*32*3*3], dconv2_b[1];
    float hyper_w[LDIM*DIM], hyper_b[LDIM];
    float hyper_inv_w[DIM*LDIM], hyper_inv_b[DIM];
    float temp_w[TEMP_DIM*DIM], temp_b[TEMP_DIM];
    float temp_merge_w[DIM*TEMP_DIM], temp_merge_b[DIM];
    float quant_scale;
} Model;

static Model model;

static int load_weights(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fread(&model, sizeof(Model), 1, f);
    fclose(f);
    return 0;
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
}

static float vec_norm(const float *x, int N) {
    float s = 0; for (int i = 0; i < N; i++) s += x[i] * x[i];
    return sqrtf(s);
}

static void poincare_clip(float *out, const float *x, int N, float c) {
    if (c <= 0.0f) { memcpy(out, x, N * sizeof(float)); return; }
    float sqrt_c = sqrtf(c + 1e-6f);
    float max_norm = 1.0f / (sqrt_c + 1e-6f);
    max_norm *= (1.0f - 1e-5f * 10.0f);
    float norm = vec_norm(x, N);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (int i = 0; i < N; i++) out[i] = x[i] * scale;
    } else {
        memcpy(out, x, N * sizeof(float));
    }
}

static void manifold_level_forward(Model* m, int l, const float* v_in, float* v_out,
                                    float* ctx_out, float* hb_out, float* hba_out) {
    int nb = m->levels[l].nb, fh = m->levels[l].fh, dim = m->levels[l].dim;
    float c = m->levels[l].c;
    float v[32]; memcpy(v, v_in, dim * sizeof(float));
    float n2 = 0; for (int i = 0; i < dim; i++) n2 += v[i]*v[i];
    float mx = 0.999f/c;
    if (n2 > mx) { float s = sqrtf(mx/(n2+1e-10f)); for (int i = 0; i < dim; i++) v[i] *= s; }
    float tv[32];
    if (n2 < 1e-15f) { memcpy(tv, v, dim * sizeof(float)); }
    else { float vn = sqrtf(n2); float c_inv = 1.0f/sqrtf(c); float arg = c_inv*vn; if (arg > 0.999f) arg = 0.999f; float ah = atanhf(arg); float sc = ah/(c_inv*vn); for (int i = 0; i < dim; i++) tv[i] = v[i]*sc; }
    float bm[32]; memset(bm, 0, sizeof(bm));
    for (int p = 0; p < nb; p++) for (int d = 0; d < dim; d++) bm[d] += m->levels[l].bd[p*dim+d];
    for (int d = 0; d < dim; d++) bm[d] /= (float)nb;
    float ctx[97]; int ci = 0;
    for (int d = 0; d < dim; d++) ctx[ci++] = tv[d];
    for (int d = 0; d < dim; d++) ctx[ci++] = bm[d];
    for (int d = 0; d < dim; d++) ctx[ci++] = m->levels[l].desc[d];
    ctx[ci++] = m->levels[l].spr;
    if (ctx_out) memcpy(ctx_out, ctx, 97*sizeof(float));
    float hb[32], hba[32], fv[32];
    for (int h = 0; h < fh; h++) { float s = m->levels[l].fb1[h]; for (int c2 = 0; c2 < 97; c2++) s += m->levels[l].fw1[h*97+c2]*ctx[c2]; hb[h] = s; hba[h] = gelu(s); }
    for (int o = 0; o < dim; o++) { float s = m->levels[l].fb2[o]; for (int h = 0; h < fh; h++) s += m->levels[l].fw2[o*fh+h]*hba[h]; fv[o] = s; }
    if (hb_out) memcpy(hb_out, hb, 32*sizeof(float));
    if (hba_out) memcpy(hba_out, hba, 32*sizeof(float));
    float to[32]; for (int i = 0; i < dim; i++) to[i] = tv[i] + fv[i];
    float ns = 0; for (int i = 0; i < dim; i++) ns += to[i]*to[i];
    if (ns > 100.0f) { float sc = 10.0f/sqrtf(ns); for (int i = 0; i < dim; i++) to[i] *= sc; }
    float vn2 = 0; for (int i = 0; i < dim; i++) vn2 += to[i]*to[i];
    float vn = sqrtf(vn2);
    if (vn < 1e-8f) { memcpy(v_out, to, dim * sizeof(float)); }
    else { float c_inv = 1.0f/sqrtf(c); float th = tanhf(c_inv*vn); float sc = th/(c_inv*vn); for (int i = 0; i < dim; i++) v_out[i] = to[i]*sc; }
    n2 = 0; for (int i = 0; i < dim; i++) n2 += v_out[i]*v_out[i];
    if (n2 > mx) { float s = sqrtf(mx/(n2+1e-10f)); for (int i = 0; i < dim; i++) v_out[i] *= s; }
}

static void conv2d(const float *input, float *output, const float *weight, const float *bias,
                   int in_ch, int out_ch, int ksize, int stride, int padding,
                   int H, int W) {
    int out_H = (H + 2*padding - ksize) / stride + 1;
    int out_W = (W + 2*padding - ksize) / stride + 1;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oy = 0; oy < out_H; oy++) {
            for (int ox = 0; ox < out_W; ox++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ky = 0; ky < ksize; ky++) {
                        for (int kx = 0; kx < ksize; kx++) {
                            int iy = oy*stride + ky - padding;
                            int ix = ox*stride + kx - padding;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                sum += weight[oc*in_ch*ksize*ksize + ic*ksize*ksize + ky*ksize + kx] *
                                       input[ic*H*W + iy*W + ix];
                            }
                        }
                    }
                }
                output[oc*out_H*out_W + oy*out_W + ox] = sum;
            }
        }
    }
}

static void deconv2d(const float *input, float *output, const float *weight, const float *bias,
                     int in_ch, int out_ch, int ksize, int stride, int padding,
                     int in_H, int in_W, int out_H, int out_W) {
    memset(output, 0, out_ch * out_H * out_W * sizeof(float));
    for (int oc = 0; oc < out_ch; oc++) {
        for (int iy = 0; iy < out_H; iy++) {
            for (int ix = 0; ix < out_W; ix++) {
                output[oc*out_H*out_W + iy*out_W + ix] = bias[oc];
            }
        }
    }
    for (int ic = 0; ic < in_ch; ic++) {
        for (int iy = 0; iy < in_H; iy++) {
            for (int ix = 0; ix < in_W; ix++) {
                float val = input[ic*in_H*in_W + iy*in_W + ix];
                for (int ky = 0; ky < ksize; ky++) {
                    for (int kx = 0; kx < ksize; kx++) {
                        int oy = iy*stride + ky - padding;
                        int ox = ix*stride + kx - padding;
                        if (oy >= 0 && oy < out_H && ox >= 0 && ox < out_W) {
                            for (int oc = 0; oc < out_ch; oc++) {
                                output[oc*out_H*out_W + oy*out_W + ox] += 
                                    val * weight[oc*in_ch*ksize*ksize + ic*ksize*ksize + ky*ksize + kx];
                            }
                        }
                    }
                }
            }
        }
    }
}

static float relu(float x) { return x > 0 ? x : 0; }

static int v92_qstep(int qp) {
    static const int t[52] = {1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722};
    if (qp < 0) qp = 0; if (qp > 51) qp = 51;
    return t[qp];
}

static double v92_psnr(const uint8_t *a, const uint8_t *b, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; m += d*d; }
    m /= n;
    return m > 0 ? 10.0 * log10(255.0 * 255.0 / m) : 99.0;
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("Usage: %s input.raw output.wubv [weights] [qp]\n", argv[0]); return 1; }

    const char *input = argv[1], *output = argv[2];
    const char *weights = argc > 3 ? argv[3] : "output/v92_trained.bin";
    int W = 64, H = 64, fps = 30, qp = 22;
    if (argc > 4) qp = atoi(argv[4]);
    
    printf("WuBuCodec v92 — Full SOTA pipeline (QP=%d)\n", qp);
    printf("  Input: %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Weights: %s\n\n", weights);

    if (load_weights(weights) != 0) { printf("Cannot load weights from %s\n", weights); return 1; }
    printf("Loaded weights\n");

    FILE *f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    int np = W * H;
    int nf = (int)(fsz / np);
    uint8_t *frames = (uint8_t *)malloc((size_t)nf * np);
    fread(frames, 1, (size_t)nf * np, f);
    fclose(f);
    printf("Loaded %d frames (%dx%d grayscale)\n", nf, W, H);

    WubuvWriter *wr = wubuv_writer_open(output, W, H, fps, nf, 0, 0, 1);
    if (!wr) { printf("Cannot create container\n"); return 1; }

    uint8_t *recon = (uint8_t *)malloc(np);
    memset(recon, 0, np);

    double total_psnr = 0;
    long total_comp = 0;

    printf("--- Encoding ---\n");
    for (int fn = 0; fn < nf; fn++) {
        const uint8_t *orig = frames + fn * np;
        int is_key = (fn % 30 == 0);

        uint8_t *fbuf = (uint8_t *)malloc(1024 * 1024);
        RangeEncoder range;
        range_init(&range, fbuf, 1024 * 1024);
        ProbModel models[16], sign_model;
        for (int i = 0; i < 16; i++) model_init(&models[i], 256, 0.05f + (float)i * 0.01f);
        model_init(&sign_model, 2, 0.1f);

        /* Conv encoder: 1x64x64 → 32x32x32 → DIMx16x16 */
        float *conv1_out = (float*)malloc(32 * 32 * 32 * sizeof(float));
        float *conv2_out = (float*)malloc(DIM * 16 * 16 * sizeof(float));
        
        conv2d(orig, conv1_out, model.conv1_w, model.conv1_b, 1, 32, 3, 2, 1, 64, 64);
        for (int i = 0; i < 32*32*32; i++) conv1_out[i] = relu(conv1_out[i]);
        conv2d(conv1_out, conv2_out, model.conv2_w, model.conv2_b, 32, DIM, 3, 2, 1, 32, 32);
        for (int i = 0; i < DIM*16*16; i++) conv2_out[i] = relu(conv2_out[i]);
        
        /* Process each 4x4 patch through manifold */
        for (int py = 0; py < 4; py++) {
            for (int px = 0; px < 4; px++) {
                float patch[16];
                for (int dy = 0; dy < 4; dy++) {
                    for (int dx = 0; dx < 4; dx++) {
                        patch[dy*4+dx] = conv2_out[((py*4+dy)*16 + (px*4+dx))];
                    }
                }
                
                float pre[32];
                for (int o = 0; o < DIM; o++) {
                    float s = model.pb[o];
                    for (int i = 0; i < 16; i++) s += model.pw[o*16+i] * patch[i];
                    pre[o] = s;
                }
                
                float mean = 0; for (int i = 0; i < DIM; i++) mean += pre[i]; mean /= DIM;
                float var = 0; for (int i = 0; i < DIM; i++) { float d = pre[i]-mean; var += d*d; } var /= DIM;
                float inv = 1/sqrtf(var + 1e-5f);
                float pl[32]; for (int i = 0; i < DIM; i++) pl[i] = (pre[i] - mean) * inv;
                
                float v[32]; memcpy(v, pl, DIM*sizeof(float));
                for (int l = 0; l < 4; l++) {
                    float vo[32];
                    manifold_level_forward(&model, l, v, vo, NULL, NULL, NULL);
                    memcpy(v, vo, DIM*sizeof(float));
                }
                
                int qs = v92_qstep(qp);
                int16_t lv[LDIM];
                for (int i = 0; i < LDIM; i++) lv[i] = (int16_t)(v[i] * 1000.0f / qs);
                
                for (int i = 0; i < LDIM; i++) {
                    int av = abs(lv[i]);
                    int sgn = (lv[i] < 0) ? 1 : 0;
                    uint32_t sym = (uint32_t)(av < 255 ? av : 255);
                    range_enc(&range, &models[i % 16], sym);
                    if (av > 0) range_enc(&range, &sign_model, sgn);
                }
                
                float dlat[LDIM];
                for (int i = 0; i < LDIM; i++) dlat[i] = (float)lv[i] * qs / 1000.0f;
                
                float v_dec[32]; memcpy(v_dec, dlat, LDIM*sizeof(float));
                
                float pl_dec[32]; for (int i = 0; i < DIM; i++) pl_dec[i] = v_dec[i];
                float pre_dec[32]; for (int i = 0; i < DIM; i++) pre_dec[i] = pl_dec[i] * sqrtf(var + 1e-5f) * inv + mean;
                
                for (int o = 0; o < 16; o++) {
                    float pred = model.db[o];
                    for (int i = 0; i < LDIM; i++) pred += model.dw[o*LDIM+i] * dlat[i];
                    int py4 = py*4 + o/4, px4 = px*4 + o%4;
                    if (py4 < 16 && px4 < 16) {
                        conv2_out[py4*16 + px4] = pred;
                    }
                }
            }
        }
        
        /* Conv decoder: DIMx16x16 → 32x32x32 → 1x64x64 */
        float *dconv1_out = (float*)malloc(32 * 32 * 32 * sizeof(float));
        float *recon_float = (float*)malloc(np * sizeof(float));
        deconv2d(conv2_out, dconv1_out, model.dconv1_w, model.dconv1_b, DIM, 32, 3, 2, 1, 16, 16, 32, 32);
        for (int i = 0; i < 32*32*32; i++) dconv1_out[i] = relu(dconv1_out[i]);
        deconv2d(dconv1_out, recon_float, model.dconv2_w, model.dconv2_b, 32, 1, 3, 2, 1, 32, 32, 64, 64);
        
        for (int i = 0; i < np; i++) {
            int v = (int)(recon_float[i] * 255.0f);
            recon[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }

        int rb = range_fin(&range);
        double p = v92_psnr(orig, recon, np);
        total_psnr += p;
        wubuv_write_video(wr, fbuf, rb, is_key);
        total_comp += rb;
        
        if (fn % 20 == 0 || fn < 5)
            printf("  Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_key ? "I" : "P", p, rb);
        
        free(conv1_out); free(conv2_out); free(dconv1_out);
        free(recon_float); free(fbuf);
    }

    wubuv_writer_close(wr);
    double avg = total_psnr / nf;
    printf("\n=== Results ===\n");
    printf("Frames: %d\n", nf);
    printf("Average PSNR: %.1f dB\n", avg);
    printf("Total: %.2f MB, bitrate: %.0f kbps\n", total_comp / 1048576.0, total_comp * 8.0 / nf * fps / 1024);
    if (wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");

    free(frames); free(recon);
    return 0;
}
