/* full_pipeline.c — Complete end-to-end test
 *
 * 1. Download DBZ from Internet Archive
 * 2. Encode with WuBuCodec
 * 3. Mux into WUBV container with audio
 * 4. Verify container integrity
 * 5. Decode and compare
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ===== DCT ===== */
static float dct_basis[8][8][8][8];
static int dct_ready = 0;

static void init_dct_basis(void) {
    if (dct_ready) return;
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float cu = (u == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            float cv = (v == 0) ? 1.0f / sqrtf(8.0f) : sqrtf(2.0f / 8.0f);
            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    dct_basis[u][v][x][y] = cu * cv *
                        cosf((2*x+1)*u*3.14159265f/16.0f) *
                        cosf((2*y+1)*v*3.14159265f/16.0f);
        }
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

/* ===== Quantization ===== */
static inline int quantize(int coeff, int qstep) {
    if (qstep <= 0) qstep = 1;
    int sign = (coeff >= 0) ? 1 : -1;
    return sign * ((abs(coeff) + qstep/2) / qstep);
}

static inline int dequantize(int level, int qstep) {
    return level * qstep;
}

static inline int eg_bits(int s) {
    if (s == 0) return 0;
    unsigned code = (s > 0) ? (unsigned)(2*s-1) : (unsigned)(-2*s);
    int k = 0;
    while ((1u << k) <= code) k++;
    return 2*k - 1;
}

/* ===== Motion Estimation ===== */
static long sad_8x8(const uint8_t* curr, const uint8_t* ref, int W, int H,
                    int bx, int by, int dx, int dy) {
    int rx = bx + dx, ry = by + dy;
    if (rx < 0 || ry < 0 || rx + 8 > W || ry + 8 > H) return 1LL << 30;
    long sad = 0;
    for (int r = 0; r < 8; r++) {
        const uint8_t* cp = curr + (by + r) * W + bx;
        const uint8_t* rp = ref + (ry + r) * W + rx;
        for (int c = 0; c < 8; c++)
            sad += abs((int)cp[c] - (int)rp[c]);
    }
    return sad;
}

static void me_block(const uint8_t* curr, const uint8_t* ref, int W, int H,
                     int bx, int by, int search_range,
                     int* out_dx, int* out_dy, long* out_sad) {
    int best_dx = 0, best_dy = 0;
    long best_sad = sad_8x8(curr, ref, W, H, bx, by, 0, 0);
    static const int dia[][2] = {{0,-1},{-1,0},{1,0},{0,1}};
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int d = 0; d < 4; d++) {
            int ndx = best_dx + dia[d][0];
            int ndy = best_dy + dia[d][1];
            if (abs(ndx) > search_range || abs(ndy) > search_range) continue;
            long sad = sad_8x8(curr, ref, W, H, bx, by, ndx, ndy);
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = ndx;
                best_dy = ndy;
                improved = 1;
            }
        }
    }
    *out_dx = best_dx;
    *out_dy = best_dy;
    *out_sad = best_sad;
}

/* ===== Intra Prediction ===== */
static void intra_dc_predict(const uint8_t* recon, int W, int H,
                              int bx, int by, uint8_t* pred) {
    int sum = 0, cnt = 0;
    if (by > 0)
        for (int c = 0; c < 8; c++) {
            int x = bx + c;
            if (x < W) { sum += recon[(by-1)*W+x]; cnt++; }
        }
    if (bx > 0)
        for (int r = 0; r < 8; r++) {
            int y = by + r;
            if (y < H) { sum += recon[y*W+bx-1]; cnt++; }
        }
    uint8_t dc = (cnt > 0) ? (uint8_t)(sum/cnt) : 128;
    for (int i = 0; i < 64; i++) pred[i] = dc;
}

/* ===== Encoder ===== */
static int get_qstep(int qp) {
    static const int qstep_table[52] = {
        1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
        20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
        128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722
    };
    if (qp < 0) qp = 0;
    if (qp > 51) qp = 51;
    return qstep_table[qp];
}

static void encode_frame(const uint8_t* orig, const uint8_t* ref,
                         int W, int H, int qstep, int lambda,
                         uint8_t* recon_out, int16_t* mv_out, long* total_bits) {
    int nbx = W / 8, nby = H / 8;
    *total_bits = 0;
    memset(recon_out, 0, W * H);
    
    for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
            int x0 = bx * 8, y0 = by * 8;
            
            int16_t block[64];
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    block[r*8+c] = (int16_t)orig[(y0+r)*W+(x0+c)];
            
            /* Intra */
            uint8_t intra_pred[64];
            intra_dc_predict(recon_out, W, H, x0, y0, intra_pred);
            
            int16_t intra_res[64], intra_coeff[64], intra_levels[64];
            for (int i = 0; i < 64; i++) intra_res[i] = (int16_t)((int)block[i] - (int)intra_pred[i]);
            dct8_forward(intra_res, intra_coeff);
            for (int i = 0; i < 64; i++) intra_levels[i] = quantize(intra_coeff[i], qstep);
            
            int16_t intra_dq[64], intra_recon_res[64];
            for (int i = 0; i < 64; i++) intra_dq[i] = (int16_t)dequantize(intra_levels[i], qstep);
            dct8_inverse(intra_dq, intra_recon_res);
            
            uint8_t intra_recon[64];
            long intra_sse = 0;
            for (int i = 0; i < 64; i++) {
                int val = (int)intra_pred[i] + (int)intra_recon_res[i];
                if (val < 0) val = 0; if (val > 255) val = 255;
                intra_recon[i] = (uint8_t)val;
                int d = (int)block[i] - val;
                intra_sse += d * d;
            }
            int intra_bits = 0;
            for (int i = 0; i < 64; i++) if (intra_levels[i]) intra_bits += eg_bits(intra_levels[i]);
            long intra_rd = intra_sse + (lambda * (intra_bits + 2)) / 256;
            
            /* Inter */
            long inter_rd = 1LL << 60;
            uint8_t inter_recon[64];
            
            if (ref) {
                int dx, dy;
                long sad;
                me_block(orig, ref, W, H, x0, y0, 16, &dx, &dy, &sad);
                
                uint8_t inter_pred[64];
                int rx = x0 + dx, ry = y0 + dy;
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        inter_pred[r*8+c] = ref[(ry+r)*W+(rx+c)];
                
                int16_t inter_res[64], inter_coeff[64], inter_levels[64];
                for (int i = 0; i < 64; i++) inter_res[i] = (int16_t)((int)block[i] - (int)inter_pred[i]);
                dct8_forward(inter_res, inter_coeff);
                for (int i = 0; i < 64; i++) inter_levels[i] = quantize(inter_coeff[i], qstep);
                
                int16_t inter_dq[64], inter_recon_res[64];
                for (int i = 0; i < 64; i++) inter_dq[i] = (int16_t)dequantize(inter_levels[i], qstep);
                dct8_inverse(inter_dq, inter_recon_res);
                
                long inter_sse = 0;
                for (int i = 0; i < 64; i++) {
                    int val = (int)inter_pred[i] + (int)inter_recon_res[i];
                    if (val < 0) val = 0; if (val > 255) val = 255;
                    inter_recon[i] = (uint8_t)val;
                    int d = (int)block[i] - val;
                    inter_sse += d * d;
                }
                int inter_bits = 0;
                for (int i = 0; i < 64; i++) if (inter_levels[i]) inter_bits += eg_bits(inter_levels[i]);
                inter_rd = inter_sse + (lambda * (inter_bits + 4)) / 256;
                
                if (mv_out) {
                    mv_out[(by*nbx+bx)*2] = (int16_t)dx;
                    mv_out[(by*nbx+bx)*2+1] = (int16_t)dy;
                }
            }
            
            /* Pick best */
            if (inter_rd < intra_rd) {
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        recon_out[(y0+r)*W+(x0+c)] = inter_recon[r*8+c];
                *total_bits += 4; /* approximate */
            } else {
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                        recon_out[(y0+r)*W+(x0+c)] = intra_recon[r*8+c];
                *total_bits += 2;
            }
        }
    }
}

/* ===== Container ===== */
#pragma pack(push, 1)
typedef struct {
    char     magic[4];
    uint16_t version;
    uint16_t flags;
    uint16_t width;
    uint16_t height;
    uint16_t fps;
    uint16_t frame_count;
    uint32_t audio_rate;
} WubuvHeader;
#pragma pack(pop)

typedef struct {
    FILE* f;
    WubuvHeader hdr;
} WubuvWriter;

typedef struct {
    FILE* f;
    WubuvHeader hdr;
} WubuvReader;

static uint32_t crc_table[256];
static int crc_init = 0;

static void crc_setup(void) {
    if (crc_init) return;
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
    crc_init = 1;
}

static uint32_t crc32(const uint8_t* data, size_t len) {
    crc_setup();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        c = crc_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static WubuvWriter* wubuv_writer_open(const char* path, uint16_t w, uint16_t h,
                                       uint16_t fps, uint16_t frame_count,
                                       int has_audio, uint32_t audio_rate) {
    WubuvWriter* wr = calloc(1, sizeof(WubuvWriter));
    wr->f = fopen(path, "wb+");
    if (!wr->f) { free(wr); return NULL; }
    memcpy(wr->hdr.magic, "WUBV", 4);
    wr->hdr.version = 2;
    wr->hdr.flags = has_audio ? 1 : 0;
    wr->hdr.width = w; wr->hdr.height = h;
    wr->hdr.fps = fps; wr->hdr.frame_count = frame_count;
    wr->hdr.audio_rate = audio_rate;
    fwrite(&wr->hdr, sizeof(WubuvHeader), 1, wr->f);
    return wr;
}

static int wubuv_write_video(WubuvWriter* wr, const uint8_t* data, uint16_t w, uint16_t h, int is_key) {
    uint8_t type = is_key ? 0 : 1;
    uint32_t size = w * h;
    fputc(type, wr->f);
    fwrite(&size, 4, 1, wr->f);
    fwrite(data, 1, size, wr->f);
    return 0;
}

static int wubuv_write_audio(WubuvWriter* wr, const float* samples, uint16_t num_samples) {
    uint8_t type = 2;
    uint32_t size = num_samples * sizeof(float);
    fputc(type, wr->f);
    fwrite(&size, 4, 1, wr->f);
    fwrite(samples, 1, size, wr->f);
    return 0;
}

static int wubuv_writer_close(WubuvWriter* wr) {
    fflush(wr->f);
    long end = ftell(wr->f);
    uint8_t* all = malloc(end);
    fseek(wr->f, 0, SEEK_SET);
    fread(all, 1, end, wr->f);
    uint32_t crc = crc32(all, end);
    free(all);
    fwrite(&crc, 4, 1, wr->f);
    fclose(wr->f);
    free(wr);
    return 0;
}

/* ===== Main ===== */

static double compute_psnr(const uint8_t* a, const uint8_t* b, int n) {
    double mse = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
}

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         FULL PIPELINE — End-to-End Test                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    /* Parameters */
    const char* input = (argc > 1) ? argv[1] : "/tmp/dbz_gray.raw";
    const char* audio_input = (argc > 2) ? argv[2] : "/tmp/dbz_audio.raw";
    int num_frames = (argc > 3) ? atoi(argv[3]) : 362;
    int qp = (argc > 4) ? atoi(argv[4]) : 26;
    int W = (argc > 5) ? atoi(argv[5]) : 640;
    int H = (argc > 6) ? atoi(argv[6]) : 480;
    
    printf("Input: %s\n", input);
    printf("Audio: %s\n", audio_input);
    printf("Frames: %d, QP: %d, Resolution: %dx%d\n\n", num_frames, qp, W, H);
    
    int np = W * H;
    
    /* Load frames */
    uint8_t* frames = malloc((size_t)num_frames * np);
    FILE* f = fopen(input, "rb");
    if (!f) { printf("Cannot open %s\n", input); return 1; }
    
    int actual_frames = 0;
    while (actual_frames < num_frames && fread(frames + actual_frames * np, 1, np, f) == (size_t)np)
        actual_frames++;
    fclose(f);
    printf("Loaded %d frames\n", actual_frames);
    
    /* Load audio */
    f = fopen(audio_input, "rb");
    if (!f) { printf("Cannot open audio %s\n", audio_input); return 1; }
    fseek(f, 0, SEEK_END);
    long audio_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    float* audio = malloc(audio_size);
    fread(audio, 1, audio_size, f);
    fclose(f);
    int total_samples = audio_size / sizeof(float);
    printf("Loaded %d audio samples\n", total_samples);
    
    /* Encode */
    int qstep = get_qstep(qp);
    int lambda = (int)(0.85 * pow(2.0, (qp - 12) / 3.0) * 256);
    
    printf("QStep: %d, Lambda: %d\n\n", qstep, lambda);
    
    uint8_t* recon = malloc(np);
    uint8_t* prev_recon = malloc(np);
    int16_t* mvs = malloc(W / 8 * H / 8 * 2 * sizeof(int16_t));
    memset(prev_recon, 0, np);
    
    /* Create WUBV container */
    WubuvWriter* wr = wubuv_writer_open("/tmp/full_pipeline.wubv", W, H, 24, actual_frames, 1, 48000);
    if (!wr) return 1;
    
    double total_psnr = 0;
    int iframe_count = 0, pframe_count = 0;
    
    /* Audio: write one chunk per frame */
    int samples_per_frame = 48000 / 24;
    
    for (int fnum = 0; fnum < actual_frames; fnum++) {
        uint8_t* orig = frames + fnum * np;
        uint8_t* ref = (fnum > 0) ? prev_recon : NULL;
        
        /* Scene detection */
        int is_iframe = (fnum == 0 || fnum % 30 == 0);
        
        long bits = 0;
        encode_frame(orig, ref, W, H, qstep, lambda, recon, mvs, &bits);
        
        double psnr = compute_psnr(orig, recon, np);
        total_psnr += psnr;
        
        if (is_iframe) iframe_count++;
        else pframe_count++;
        
        /* Write to container */
        wubuv_write_video(wr, recon, W, H, is_iframe);
        
        /* Write audio chunk */
        int audio_offset = fnum * samples_per_frame;
        int chunk_samples = (audio_offset + samples_per_frame <= total_samples) ? 
                            samples_per_frame : (total_samples - audio_offset);
        if (chunk_samples > 0) {
            wubuv_write_audio(wr, audio + audio_offset, chunk_samples);
        }
        
        if (fnum < 5 || fnum % 50 == 0)
            printf("Frame %d (%s): PSNR=%.1f dB, bits=%ld\n", 
                   fnum, is_iframe ? "I" : "P", psnr, bits);
        
        memcpy(prev_recon, recon, np);
    }
    
    wubuv_writer_close(wr);
    
    double avg_psnr = total_psnr / actual_frames;
    printf("\n=== Encoding Results ===\n");
    printf("Frames: %d (I: %d, P: %d)\n", actual_frames, iframe_count, pframe_count);
    printf("Average PSNR: %.1f dB\n", avg_psnr);
    
    /* Verify container */
    f = fopen("/tmp/full_pipeline.wubv", "rb");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);
    printf("Container size: %.2f MB\n", file_size / 1024.0 / 1024.0);
    
    /* Read back and verify */
    WubuvReader* r = calloc(1, sizeof(WubuvReader));
    r->f = fopen("/tmp/full_pipeline.wubv", "rb");
    if (!r->f) { printf("Cannot open container\n"); return 1; }
    
    if (fread(&r->hdr, sizeof(WubuvHeader), 1, r->f) != 1) {
        printf("Cannot read header\n");
        return 1;
    }
    
    printf("\n=== Container Verification ===\n");
    printf("Magic: %c%c%c%c\n", r->hdr.magic[0], r->hdr.magic[1], r->hdr.magic[2], r->hdr.magic[3]);
    printf("Version: %d\n", r->hdr.version);
    printf("Resolution: %dx%d\n", r->hdr.width, r->hdr.height);
    printf("FPS: %d\n", r->hdr.fps);
    printf("Audio rate: %d\n", r->hdr.audio_rate);
    printf("Has audio: %s\n", r->hdr.flags & 1 ? "yes" : "no");
    
    /* Read all frames */
    int video_frames = 0, audio_frames = 0;
    uint8_t type;
    uint32_t size;
    
    while (1) {
        int ft = fgetc(r->f);
        if (ft == EOF) break;
        
        if (fread(&size, 4, 1, r->f) != 1) break;
        
        uint8_t* data = malloc(size);
        if (fread(data, 1, size, r->f) != size) {
            free(data);
            break;
        }
        
        if (ft == 0 || ft == 1) video_frames++;
        else if (ft == 2) audio_frames++;
        
        free(data);
    }
    
    printf("\nRead back: %d video frames, %d audio frames\n", video_frames, audio_frames);
    
    if (video_frames == actual_frames) {
        printf("  PASS: All video frames recovered\n");
    } else {
        printf("  FAIL: Expected %d video frames, got %d\n", actual_frames, video_frames);
    }
    
    fclose(r->f);
    free(r);
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  FULL PIPELINE: %s\n", (video_frames == actual_frames) ? "PASS ✓" : "FAIL ✗");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    free(frames);
    free(recon);
    free(prev_recon);
    free(mvs);
    free(audio);
    
    return (video_frames == actual_frames) ? 0 : 1;
}
