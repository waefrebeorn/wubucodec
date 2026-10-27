/* wubuv_encode.c — Half-pel ME + MV cost modeling + test on DBZ */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

/* ===== DCT ===== */
static float dct_basis[8][8][8][8];
static int dct_ready = 0;

static void init_dct_basis(void) {
    if (dct_ready) return;
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) {
            float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    dct_basis[u][v][x][y] = cu * cv *
                        cosf((2*x+1)*u*3.14159265f/16.0f) *
                        cosf((2*y+1)*v*3.14159265f/16.0f);
        }
    dct_ready = 1;
}

static void dct8_forward(const int16_t* in, int16_t* out) {
    init_dct_basis();
    for (int u = 0; u < 8; u++)
        for (int v = 0; v < 8; v++) {
            float sum = 0;
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    sum += in[x*8+y] * dct_basis[u][v][x][y];
            out[u*8+v] = (int16_t)roundf(sum);
        }
}

static void dct8_inverse(const int16_t* in, int16_t* out) {
    init_dct_basis();
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) {
            float sum = 0;
            for (int u = 0; u < 8; u++)
                for (int v = 0; v < 8; v++)
                    sum += in[u*8+v] * dct_basis[u][v][x][y];
            out[x*8+y] = (int16_t)roundf(sum);
        }
}

static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep/2) / qstep);
}

static inline int dequantize(int level, int qstep) {
    return level * qstep;
}

static const int zigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

/* ===== Bitstream ===== */
typedef struct {
    uint8_t* buf;
    int byte_pos;
    int bit_pos;
    int size;
} BitStream;

static void bs_init(BitStream* bs, uint8_t* buf, int size) {
    bs->buf = buf; bs->byte_pos = 0; bs->bit_pos = 0; bs->size = size;
}

static void bs_write_bit(BitStream* bs, int bit) {
    if (bs->byte_pos >= bs->size) return;
    if (bit) bs->buf[bs->byte_pos] |= (1 << (7 - bs->bit_pos));
    bs->bit_pos++;
    if (bs->bit_pos == 8) { bs->bit_pos = 0; bs->byte_pos++; }
}

static void bs_write_ue(BitStream* bs, int val) {
    int p = 0, t = val;
    while (t > 0) { bs_write_bit(bs, 1); t--; p++; }
    bs_write_bit(bs, 0);
    if (p > 0) for (int i = p-1; i >= 0; i--) bs_write_bit(bs, (val >> i) & 1);
}

static void bs_write_se(BitStream* bs, int val) {
    int a = abs(val), s = (val < 0) ? 1 : 0;
    bs_write_ue(bs, a);
    if (a > 0) bs_write_bit(bs, s);
}

static int bs_bytes(BitStream* bs) {
    return bs->byte_pos + (bs->bit_pos > 0 ? 1 : 0);
}

/* ===== CABAC-like coding ===== */
static int encode_block(int16_t* levels, uint8_t* out, int sz) {
    BitStream bs; bs_init(&bs, out, sz);
    int nz = 0;
    for (int i = 0; i < 64; i++) if (levels[zigzag[i]] != 0) nz++;
    bs_write_ue(&bs, nz);
    if (nz == 0) return bs_bytes(&bs);
    int run = 0;
    for (int i = 0; i < 64; i++) {
        int zz = zigzag[i], lv = levels[zz];
        if (lv != 0) { bs_write_ue(&bs, run); bs_write_se(&bs, lv); run = 0; }
        else run++;
    }
    return bs_bytes(&bs);
}

static int block_var(const int16_t* b) {
    int s = 0, s2 = 0;
    for (int i = 0; i < 64; i++) { s += b[i]; s2 += b[i]*b[i]; }
    return s2/64 - (s/64)*(s/64);
}

static int get_qstep(int qp) {
    static const int t[52] = {
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722
    };
    if (qp < 0) qp = 0; if (qp > 51) qp = 51;
    return t[qp];
}

static int aq_qp(int base, int var) {
    if (var < 100) return base - 4;
    if (var < 500) return base - 2;
    if (var < 2000) return base;
    if (var < 5000) return base + 2;
    return base + 4;
}

/* ===== Motion Estimation with Half-Pel ===== */

static long sad_8x8(const uint8_t* cur, const uint8_t* ref, int W, int H, int bx, int by, int dx, int dy) {
    int rx = bx + dx, ry = by + dy;
    if (rx < 0 || ry < 0 || rx+8 > W || ry+8 > H) return 1LL << 30;
    long s = 0;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            s += abs((int)cur[(by+r)*W+bx+c] - (int)ref[(ry+r)*W+rx+c]);
    return s;
}

/* Integer-pel ME with diamond search */
static void me_int(const uint8_t* cur, const uint8_t* ref, int W, int H,
                   int bx, int by, int sr, int* dx, int* dy, long* sad) {
    *dx = 0; *dy = 0; *sad = sad_8x8(cur, ref, W, H, bx, by, 0, 0);
    static const int d[4][2] = {{0,-1},{-1,0},{1,0},{0,1}};
    int imp = 1;
    while (imp) {
        imp = 0;
        for (int i = 0; i < 4; i++) {
            int nx = *dx + d[i][0], ny = *dy + d[i][1];
            if (abs(nx) > sr || abs(ny) > sr) continue;
            long s = sad_8x8(cur, ref, W, H, bx, by, nx, ny);
            if (s < *sad) { *sad = s; *dx = nx; *dy = ny; imp = 1; }
        }
    }
}

/* Half-pel interpolation: generate 8x8 block at half-pel position */
static void half_pel_interp(const uint8_t* ref, int W, int H,
                           int bx, int by, int hdx, int hdy, uint8_t* out) {
    /* hdx, hdy are in half-pel units (2*integer + {-1,0,1}) */
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            /* Position in reference at half-pel resolution */
            int x2 = (bx + c) * 2 + hdx;
            int y2 = (by + r) * 2 + hdy;
            
            /* Integer coords */
            int x0 = x2 / 2, y0 = y2 / 2;
            int x1 = x0 + 1, y1 = y0 + 1;
            int fx = x2 & 1, fy = y2 & 1;
            
            /* Clamp */
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
            if (x1 >= W) x1 = W-1; if (y1 >= H) y1 = H-1;
            
            /* Bilinear interpolate */
            int v = (ref[y0*W+x0] * (2-fx) * (2-fy) +
                     ref[y0*W+x1] * fx * (2-fy) +
                     ref[y1*W+x0] * (2-fx) * fy +
                     ref[y1*W+x1] * fx * fy) / 4;
            out[r*8+c] = (uint8_t)v;
        }
    }
}

/* Half-pel motion estimation: refine integer MV to half-pel precision */
static void me_half_pel(const uint8_t* cur, const uint8_t* ref, int W, int H,
                        int bx, int by, int sr,
                        int* out_hdx, int* out_hdy, long* out_sad) {
    /* First find integer MV */
    int idx, idy, isad;
    me_int(cur, ref, W, H, bx, by, sr, &idx, &idy, &isad);
    
    /* Convert to half-pel units (multiply by 2) */
    int best_hdx = idx * 2;
    int best_hdy = idy * 2;
    long best_sad = isad;
    
    uint8_t hp_block[64];
    
    /* Search 3x3 half-pel window around integer MV */
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int thdx = idx * 2 + dx;
            int thdy = idy * 2 + dy;
            
            half_pel_interp(ref, W, H, bx, by, thdx, thdy, hp_block);
            
            long s = 0;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    s += abs((int)cur[(by+r)*W+bx+c] - (int)hp_block[r*8+c]);
            
            if (s < best_sad) {
                best_sad = s;
                best_hdx = thdx;
                best_hdy = thdy;
            }
        }
    }
    
    *out_hdx = best_hdx;
    *out_hdy = best_hdy;
    *out_sad = best_sad;
}

/* Get motion-compensated prediction block from half-pel MV */
static void mc_get_pred(const uint8_t* ref, int W, int H,
                        int bx, int by, int hdx, int hdy,
                        uint8_t* pred) {
    half_pel_interp(ref, W, H, bx, by, hdx, hdy, pred);
}

/* Estimate MV coding cost (bits for exp-golomb of dx, dy) */
static int mv_cost(int dx, int dy) {
    /* Convert half-pel to integer for coding (divide by 2) */
    int mx = dx / 2;
    int my = dy / 2;
    if (dx & 1) mx += (dx < 0 ? -1 : 1); /* Round */
    if (dy & 1) my += (dy < 0 ? -1 : 1);
    
    int bits = 0;
    /* Exp-golomb bits for mx */
    int code = (mx >= 0) ? 2*mx : -2*mx-1;
    int k = 0;
    while ((1 << k) <= code) k++;
    bits += (code > 0) ? 2*k - 1 : 1;
    /* Exp-golomb bits for my */
    code = (my >= 0) ? 2*my : -2*my-1;
    k = 0;
    while ((1 << k) <= code) k++;
    bits += (code > 0) ? 2*k - 1 : 1;
    
    return bits;
}

/* ===== Intra Prediction ===== */
static void intra_dc(const uint8_t* rec, int W, int H, int bx, int by, uint8_t* p) {
    int s = 0, n = 0;
    if (by > 0) for (int c = 0; c < 8; c++) { int x = bx+c; if (x < W) { s += rec[(by-1)*W+x]; n++; } }
    if (bx > 0) for (int r = 0; r < 8; r++) { int y = by+r; if (y < H) { s += rec[y*W+bx-1]; n++; } }
    uint8_t dc = (n > 0) ? (uint8_t)(s/n) : 128;
    for (int i = 0; i < 64; i++) p[i] = dc;
}

static void intra_vert(const uint8_t* rec, int W, int H, int bx, int by, uint8_t* p) {
    if (by == 0) { intra_dc(rec, W, H, bx, by, p); return; }
    for (int c = 0; c < 8; c++) {
        int x = bx+c; if (x >= W) break;
        uint8_t v = rec[(by-1)*W+x];
        for (int r = 0; r < 8; r++) p[r*8+c] = v;
    }
}

static void intra_horiz(const uint8_t* rec, int W, int H, int bx, int by, uint8_t* p) {
    if (bx == 0) { intra_dc(rec, W, H, bx, by, p); return; }
    for (int r = 0; r < 8; r++) {
        int y = by+r; if (y >= H) break;
        uint8_t v = rec[y*W+bx-1];
        for (int c = 0; c < 8; c++) p[r*8+c] = v;
    }
}

/* ===== PSNR ===== */
static double psnr(const uint8_t* a, const uint8_t* b, int n) {
    double mse = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d*d; }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

/* ===== Main ===== */
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s input.raw output.wubv [options]\n", argv[0]);
        printf("  -w width      Video width (default 320)\n");
        printf("  -h height     Video height (default 240)\n");
        printf("  -f fps        Frames per second (default 30)\n");
        printf("  -q qp         Quantization parameter 0-51 (default 26)\n");
        printf("  -a audio.raw  Optional raw float32 PCM audio file\n");
        printf("  -r rate       Audio sample rate (default 48000)\n");
        return 1;
    }
    
    const char* input = argv[1];
    const char* output = argv[2];
    int W = 320, H = 240, fps = 30, qp = 26;
    const char* audio_file = NULL;
    int audio_rate = 48000;
    
    for (int i = 3; i < argc; i += 2) {
        if (i+1 >= argc) break;
        if (strcmp(argv[i], "-w") == 0) W = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-h") == 0) H = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-f") == 0) fps = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-q") == 0) qp = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-a") == 0) audio_file = argv[i+1];
        else if (strcmp(argv[i], "-r") == 0) audio_rate = atoi(argv[i+1]);
    }
    
    int np = W * H;
    
    printf("WUBV Encoder v6 (Half-pel ME + MV cost)\n");
    printf("  Input:  %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Resolution: %dx%d @ %d fps, QP=%d\n", W, H, fps, qp);
    if (audio_file) printf("  Audio: %s @ %d Hz\n", audio_file, audio_rate);
    printf("\n");
    
    FILE* f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int nf = fsz / np;
    uint8_t* frames = malloc((size_t)nf * np);
    if (fread(frames, 1, (size_t)nf*np, f) != (size_t)(nf*np)) { printf("Read failed\n"); return 1; }
    fclose(f);
    printf("Loaded %d frames\n", nf);
    
    float* audio = NULL;
    int total_samp = 0;
    if (audio_file) {
        f = fopen(audio_file, "rb");
        if (!f) { printf("Cannot open audio\n"); return 1; }
        fseek(f, 0, SEEK_END); long asz = ftell(f); fseek(f, 0, SEEK_SET);
        total_samp = asz / sizeof(float);
        audio = malloc(asz);
        fread(audio, 1, asz, f);
        fclose(f);
    }
    
    WubuvWriter* wr = wubuv_writer_open(output, W, H, fps, nf, audio != NULL, audio_rate, 1);
    if (!wr) { printf("Cannot create container\n"); return 1; }
    
    uint8_t* recon = malloc(np);
    uint8_t* prev_recon = malloc(np);
    memset(prev_recon, 0, np);
    
    double total_psnr = 0;
    int ifr = 0, pfr = 0, bfr = 0;
    long total_comp = 0;
    int inter_wins = 0, intra_wins = 0;
    int spf = audio_rate / fps;
    
    for (int fn = 0; fn < nf; fn++) {
        uint8_t* orig = frames + fn * np;
        uint8_t* ref = (fn > 0) ? prev_recon : NULL;
        int is_i = (fn == 0 || fn % 30 == 0);
        
        int nbx = W/8, nby = H/8;
        uint8_t* fbuf = malloc(nbx*nby*256);
        int fpos = 0;
        
        for (int by = 0; by < nby; by++) {
            for (int bx = 0; bx < nbx; bx++) {
                int x0 = bx*8, y0 = by*8;
                
                int16_t blk[64];
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        blk[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];
                
                int best_sz = 1 << 30;
                uint8_t best_pred[64];
                int16_t best_lv[64];
                int best_qs = get_qstep(qp);
                
                /* Try INTER prediction if we have a reference */
                if (ref && !is_i) {
                    int hdx, hdy;
                    long sad;
                    me_half_pel(orig, ref, W, H, x0, y0, 16, &hdx, &hdy, &sad);
                    
                    uint8_t inter_pred[64];
                    mc_get_pred(ref, W, H, x0, y0, hdx, hdy, inter_pred);
                    
                    int16_t res[64], coeff[64], lv[64];
                    for (int i = 0; i < 64; i++)
                        res[i] = (int16_t)((int)blk[i] - (int)inter_pred[i]);
                    
                    int var = block_var(res);
                    int qs = get_qstep(aq_qp(qp, var));
                    dct8_forward(res, coeff);
                    for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                    
                    /* Total cost = MV bits + residual bits */
                    int mv_bits = mv_cost(hdx, hdy);
                    uint8_t bb[256];
                    int coef_sz = encode_block(lv, bb, 256);
                    int total_inter = coef_sz + mv_bits / 8;
                    
                    if (total_inter < best_sz) {
                        best_sz = total_inter;
                        memcpy(best_pred, inter_pred, 64);
                        memcpy(best_lv, lv, sizeof(lv));
                        best_qs = qs;
                    }
                }
                
                /* Try INTRA predictions */
                uint8_t dc_p[64], vt_p[64], hz_p[64];
                intra_dc(recon, W, H, x0, y0, dc_p);
                intra_vert(recon, W, H, x0, y0, vt_p);
                intra_horiz(recon, W, H, x0, y0, hz_p);
                
                int16_t res[64], coeff[64], lv[64];
                
                /* DC */
                for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)dc_p[i]);
                dct8_forward(res, coeff);
                int var = block_var(res);
                int qs = get_qstep(aq_qp(qp, var));
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                uint8_t bb[256];
                int dc_sz = encode_block(lv, bb, 256);
                if (dc_sz < best_sz) {
                    best_sz = dc_sz;
                    memcpy(best_pred, dc_p, 64);
                    memcpy(best_lv, lv, sizeof(lv));
                    best_qs = qs;
                }
                
                /* Vertical */
                for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)vt_p[i]);
                dct8_forward(res, coeff);
                var = block_var(res);
                qs = get_qstep(aq_qp(qp, var));
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                int vt_sz = encode_block(lv, bb, 256);
                if (vt_sz < best_sz) {
                    best_sz = vt_sz;
                    memcpy(best_pred, vt_p, 64);
                    memcpy(best_lv, lv, sizeof(lv));
                    best_qs = qs;
                }
                
                /* Horizontal */
                for (int i = 0; i < 64; i++) res[i] = (int16_t)((int)blk[i] - (int)hz_p[i]);
                dct8_forward(res, coeff);
                var = block_var(res);
                qs = get_qstep(aq_qp(qp, var));
                for (int i = 0; i < 64; i++) lv[i] = quantize(coeff[i], qs);
                int hz_sz = encode_block(lv, bb, 256);
                if (hz_sz < best_sz) {
                    best_sz = hz_sz;
                    memcpy(best_pred, hz_p, 64);
                    memcpy(best_lv, lv, sizeof(lv));
                    best_qs = qs;
                }
                
                /* Encode best */
                int bsz = encode_block(best_lv, bb, 256);
                memcpy(fbuf + fpos, bb, bsz);
                fpos += bsz;
                
                /* Reconstruct */
                int16_t dq[64], rres[64];
                for (int i = 0; i < 64; i++) dq[i] = (int16_t)dequantize(best_lv[i], best_qs);
                dct8_inverse(dq, rres);
                
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++) {
                        int v = (int)best_pred[r*8+c] + (int)rres[r*8+c];
                        if (v < 0) v = 0; if (v > 255) v = 255;
                        recon[(y0+r)*W+(x0+c)] = (uint8_t)v;
                    }
            }
        }
        
        double p = psnr(orig, recon, np);
        total_psnr += p;
        if (is_i) ifr++; else pfr++;
        
        wubuv_write_video(wr, fbuf, fpos, is_i);
        total_comp += fpos;
        
        if (audio) {
            int off = fn * spf;
            int chunk = (off + spf <= total_samp) ? spf : (total_samp - off);
            if (chunk > 0) wubuv_write_audio(wr, audio + off, chunk);
        }
        
        if (fn % 50 == 0 || fn < 5)
            printf("  Frame %d (%s): PSNR=%.1f dB, bytes=%d\n", fn, is_i ? "I" : "P", p, fpos);
        
        memcpy(prev_recon, recon, np);
        free(fbuf);
    }
    
    wubuv_writer_close(wr);
    double avg = total_psnr / nf;
    
    printf("\n=== Results ===\n");
    printf("Frames: %d (I:%d P:%d)\n", nf, ifr, pfr);
    printf("Average PSNR: %.1f dB\n", avg);
    printf("Total: %.2f MB, bitrate: %ld kbps\n",
           total_comp / 1048576.0, total_comp * 8 / 10 / 1024);
    if (wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");
    
    free(frames); free(recon); free(prev_recon);
    if (audio) free(audio);
    return 0;
}
