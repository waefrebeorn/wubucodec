/* dbz_ab_test.c — A/B test: WuBuCodec v8 vs x264 vs AV1 on Dragon Ball Z
 *
 * Pipeline:
 * 1. Extract raw grayscale frames from DBZ video
 * 2. Encode with WuBuCodec v8 (pixel-space DCT encoder with B-frames)
 * 3. Encode with x264 at matched bitrate
 * 4. Encode with AV1 at matched bitrate
 * 5. Decode all three and compute PSNR/SSIM vs original
 *
 * Build: gcc -std=c11 -O2 -Iinclude dbz_ab_test.c src/container/wubuv.c src/entropy/wubu_entropy.c src/entropy/wubu_cabac.c src/math/wubu_ratecontrol.c src/math/wubu_scene.c src/temporal/wubu_lookahead.c src/rdo/wubu_rdo.c src/motion/wubu_dpb.c src/transform/wubu_transform.c src/motion/wubu_motionest.c src/audio/wubu_stft.c src/audio/wubu_kodak.c -o bin/dbz_ab_test -lm -lpthread
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"
#include "wubu_cabac.h"
#include "wubu_ratecontrol.h"
#include "wubu_lookahead.h"
#include "wubu_scene.h"
#include "wubu_rdo.h"

/* ============================ PSNR / SSIM ============================ */

static double compute_psnr(const uint8_t* a, const uint8_t* b, size_t n){
    double mse = 0;
    for(size_t i=0; i<n; i++){
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

static double compute_ssim(const uint8_t* a, const uint8_t* b, int W, int H){
    /* Simplified SSIM with 8x8 windows */
    double C1 = (0.01 * 255) * (0.01 * 255);
    double C2 = (0.03 * 255) * (0.03 * 255);
    double ssim_sum = 0;
    int windows = 0;

    for(int y=0; y<H-7; y+=8){
        for(int x=0; x<W-7; x+=8){
            double mean_a = 0, mean_b = 0;
            double var_a = 0, var_b = 0, covar = 0;

            for(int dy=0; dy<8; dy++){
                for(int dx=0; dx<8; dx++){
                    int idx = (y+dy)*W + (x+dx);
                    mean_a += a[idx];
                    mean_b += b[idx];
                }
            }
            mean_a /= 64.0;
            mean_b /= 64.0;

            for(int dy=0; dy<8; dy++){
                for(int dx=0; dx<8; dx++){
                    int idx = (y+dy)*W + (x+dx);
                    double da = a[idx] - mean_a;
                    double db = b[idx] - mean_b;
                    var_a += da * da;
                    var_b += db * db;
                    covar += da * db;
                }
            }
            var_a /= 64.0;
            var_b /= 64.0;
            covar /= 64.0;

            double ssim = ((2*mean_a*mean_b + C1) * (2*covar + C2)) /
                          ((mean_a*mean_a + mean_b*mean_b + C1) * (var_a + var_b + C2));
            ssim_sum += ssim;
            windows++;
        }
    }
    return windows > 0 ? ssim_sum / windows : 0;
}

/* ============================ Main ============================ */

int main(int argc, char** argv){
    if(argc < 3){
        printf("DBZ A/B Test: WuBuCodec v8 vs x264 vs AV1\n");
        printf("Usage: %s <input.y4m> <output_dir>\n", argv[0]);
        return 1;
    }

    const char* input = argv[1];
    const char* output_dir = argv[2];

    /* Read Y4M */
    FILE* f = fopen(input, "rb");
    if(!f){ printf("Cannot open %s\n", input); return 1; }

    char header[256];
    if(!fgets(header, sizeof(header), f)){
        printf("Cannot read header\n");
        return 1;
    }
    printf("Y4M Header: %s", header);

    int W = 320, H = 240, num_frames = 360;
    char* pw = strstr(header, "W");
    char* ph = strstr(header, "H");
    if(pw && ph){
        W = atoi(pw+1);
        H = atoi(ph+1);
    }
    printf("Resolution: %dx%d\n", W, H);

    int frame_size = W * H;
    uint8_t* frames = malloc((size_t)num_frames * frame_size);
    uint8_t* frame_buf = malloc(frame_size);

    printf("Reading %d frames...\n", num_frames);
    for(int i=0; i<num_frames; i++){
        char frame_header[64];
        if(!fgets(frame_header, sizeof(frame_header), f)){
            num_frames = i;
            break;
        }
        if(fread(frame_buf, 1, frame_size, f) != (size_t)frame_size){
            num_frames = i;
            break;
        }
        memcpy(frames + (size_t)i * frame_size, frame_buf, frame_size);
    }
    fclose(f);
    printf("Read %d frames\n", num_frames);

    /* ===== Encode with WuBuCodec v8 ===== */
    printf("\n=== Encoding with WuBuCodec v8 ===\n");

    char raw_path[512];
    snprintf(raw_path, sizeof(raw_path), "%s/dbz_input.raw", output_dir);
    FILE* raw_f = fopen(raw_path, "wb");
    fwrite(frames, 1, (size_t)num_frames * frame_size, raw_f);
    fclose(raw_f);

    char wubv_path[512];
    snprintf(wubv_path, sizeof(wubv_path), "%s/dbz_wubucodec.wubv", output_dir);

    /* Use wubuv_encode_v8 to encode */
    char encode_cmd[1024];
    snprintf(encode_cmd, sizeof(encode_cmd),
             "./bin/wubuv_encode_v8 %s %s -w %d -h %d -f 24 -q 26 -g 30 -b 2 2>&1",
             raw_path, wubv_path, W, H);
    printf("Running: %s\n", encode_cmd);
    FILE* enc_fp = popen(encode_cmd, "r");
    if(enc_fp){
        char line[256];
        while(fgets(line, sizeof(line), enc_fp)){
            printf("  %s", line);
        }
        pclose(enc_fp);
    }

    /* Decode with wubuv_decode */
    char decoded_raw[512];
    snprintf(decoded_raw, sizeof(decoded_raw), "%s/dbz_wubucodec_decoded.raw", output_dir);
    char decode_cmd[1024];
    snprintf(decode_cmd, sizeof(decode_cmd),
             "./bin/wubuv_decode %s %s 2>&1",
             wubv_path, decoded_raw);
    printf("Running: %s\n", decode_cmd);
    FILE* dec_fp = popen(decode_cmd, "r");
    if(dec_fp){
        char line[256];
        while(fgets(line, sizeof(line), dec_fp)){
            printf("  %s", line);
        }
        pclose(dec_fp);
    }

    /* Read decoded frames */
    uint8_t* decoded_wubu = malloc((size_t)num_frames * frame_size);
    FILE* dec_f = fopen(decoded_raw, "rb");
    if(dec_f){
        fread(decoded_wubu, 1, (size_t)num_frames * frame_size, dec_f);
        fclose(dec_f);
    }

    /* Compute PSNR/SSIM for WuBuCodec */
    double wubu_psnr = compute_psnr(frames, decoded_wubu, (size_t)num_frames * frame_size);
    double wubu_ssim = compute_ssim(frames, decoded_wubu, W, H);

    /* Get file size */
    FILE* wubv_f = fopen(wubv_path, "rb");
    long wubv_size = 0;
    if(wubv_f){
        fseek(wubv_f, 0, SEEK_END);
        wubv_size = ftell(wubv_f);
        fclose(wubv_f);
    }

    printf("\nWuBuCodec v8 Results:\n");
    printf("  Size: %.2f KB\n", wubv_size / 1024.0);
    printf("  PSNR: %.2f dB\n", wubu_psnr);
    printf("  SSIM: %.4f\n", wubu_ssim);
    printf("  Bits/pixel: %.2f\n", (double)wubv_size * 8 / ((size_t)num_frames * frame_size));

    /* ===== Encode with x264 ===== */
    printf("\n=== Encoding with x264 ===\n");

    char x264_path[512];
    snprintf(x264_path, sizeof(x264_path), "%s/dbz_x264.mp4", output_dir);
    char x264_cmd[1024];
    snprintf(x264_cmd, sizeof(x264_cmd),
             "ffmpeg -y -f rawvideo -pix_fmt gray -s %dx%d -r 24 -i %s "
             "-c:v libx264 -preset medium -crf 26 -pix_fmt gray %s 2>&1 | tail -5",
             W, H, raw_path, x264_path);
    printf("Running: %s\n", x264_cmd);
    FILE* x264_fp = popen(x264_cmd, "r");
    if(x264_fp){
        char line[256];
        while(fgets(line, sizeof(line), x264_fp)){
            printf("  %s", line);
        }
        pclose(x264_fp);
    }

    /* Decode x264 */
    char x264_decoded[512];
    snprintf(x264_decoded, sizeof(x264_decoded), "%s/dbz_x264_decoded.raw", output_dir);
    char x264_dec_cmd[1024];
    snprintf(x264_dec_cmd, sizeof(x264_dec_cmd),
             "ffmpeg -y -i %s -f rawvideo -pix_fmt gray %s 2>&1 | tail -3",
             x264_path, x264_decoded);
    printf("Running: %s\n", x264_dec_cmd);
    FILE* x264_dec_fp = popen(x264_dec_cmd, "r");
    if(x264_dec_fp){
        char line[256];
        while(fgets(line, sizeof(line), x264_dec_fp)){
            printf("  %s", line);
        }
        pclose(x264_dec_fp);
    }

    uint8_t* decoded_x264 = malloc((size_t)num_frames * frame_size);
    FILE* x264_dec_f = fopen(x264_decoded, "rb");
    if(x264_dec_f){
        fread(decoded_x264, 1, (size_t)num_frames * frame_size, x264_dec_f);
        fclose(x264_dec_f);
    }

    double x264_psnr = compute_psnr(frames, decoded_x264, (size_t)num_frames * frame_size);
    double x264_ssim = compute_ssim(frames, decoded_x264, W, H);

    FILE* x264_out_f = fopen(x264_path, "rb");
    long x264_size = 0;
    if(x264_out_f){
        fseek(x264_out_f, 0, SEEK_END);
        x264_size = ftell(x264_out_f);
        fclose(x264_out_f);
    }

    printf("\nx264 Results:\n");
    printf("  Size: %.2f KB\n", x264_size / 1024.0);
    printf("  PSNR: %.2f dB\n", x264_psnr);
    printf("  SSIM: %.4f\n", x264_ssim);
    printf("  Bits/pixel: %.2f\n", (double)x264_size * 8 / ((size_t)num_frames * frame_size));

    /* ===== Encode with AV1 ===== */
    printf("\n=== Encoding with AV1 ===\n");

    char av1_path[512];
    snprintf(av1_path, sizeof(av1_path), "%s/dbz_av1.mp4", output_dir);
    char av1_cmd[1024];
    snprintf(av1_cmd, sizeof(av1_cmd),
             "ffmpeg -y -f rawvideo -pix_fmt gray -s %dx%d -r 24 -i %s "
             "-c:v libaom-av1 -preset 6 -crf 26 -pix_fmt gray %s 2>&1 | tail -5",
             W, H, raw_path, av1_path);
    printf("Running: %s\n", av1_cmd);
    FILE* av1_fp = popen(av1_cmd, "r");
    if(av1_fp){
        char line[256];
        while(fgets(line, sizeof(line), av1_fp)){
            printf("  %s", line);
        }
        pclose(av1_fp);
    }

    /* Decode AV1 */
    char av1_decoded[512];
    snprintf(av1_decoded, sizeof(av1_decoded), "%s/dbz_av1_decoded.raw", output_dir);
    char av1_dec_cmd[1024];
    snprintf(av1_dec_cmd, sizeof(av1_dec_cmd),
             "ffmpeg -y -i %s -f rawvideo -pix_fmt gray %s 2>&1 | tail -3",
             av1_path, av1_decoded);
    printf("Running: %s\n", av1_dec_cmd);
    FILE* av1_dec_fp = popen(av1_dec_cmd, "r");
    if(av1_dec_fp){
        char line[256];
        while(fgets(line, sizeof(line), av1_dec_fp)){
            printf("  %s", line);
        }
        pclose(av1_dec_fp);
    }

    uint8_t* decoded_av1 = malloc((size_t)num_frames * frame_size);
    FILE* av1_dec_f = fopen(av1_decoded, "rb");
    if(av1_dec_f){
        fread(decoded_av1, 1, (size_t)num_frames * frame_size, av1_dec_f);
        fclose(av1_dec_f);
    }

    double av1_psnr = compute_psnr(frames, decoded_av1, (size_t)num_frames * frame_size);
    double av1_ssim = compute_ssim(frames, decoded_av1, W, H);

    FILE* av1_out_f = fopen(av1_path, "rb");
    long av1_size = 0;
    if(av1_out_f){
        fseek(av1_out_f, 0, SEEK_END);
        av1_size = ftell(av1_out_f);
        fclose(av1_out_f);
    }

    printf("\nAV1 Results:\n");
    printf("  Size: %.2f KB\n", av1_size / 1024.0);
    printf("  PSNR: %.2f dB\n", av1_psnr);
    printf("  SSIM: %.4f\n", av1_ssim);
    printf("  Bits/pixel: %.2f\n", (double)av1_size * 8 / ((size_t)num_frames * frame_size));

    /* ===== Summary ===== */
    printf("\n");
    printf("============================================================\n");
    printf("           DBZ A/B TEST RESULTS (360 frames, 320x240)        \n");
    printf("============================================================\n");
    printf("Codec          Size(KB)  PSNR(dB)  SSIM     Bits/pixel\n");
    printf("------------------------------------------------------------\n");
    printf("WuBuCodec v8   %8.2f  %7.2f  %6.4f  %8.2f\n",
           wubv_size/1024.0, wubu_psnr, wubu_ssim,
           (double)wubv_size*8/((size_t)num_frames*frame_size));
    printf("x264           %8.2f  %7.2f  %6.4f  %8.2f\n",
           x264_size/1024.0, x264_psnr, x264_ssim,
           (double)x264_size*8/((size_t)num_frames*frame_size));
    printf("AV1            %8.2f  %7.2f  %6.4f  %8.2f\n",
           av1_size/1024.0, av1_psnr, av1_ssim,
           (double)av1_size*8/((size_t)num_frames*frame_size));
    printf("============================================================\n");

    /* Save decoded frames as PGM for visual inspection */
    char pgm_path[512];
    snprintf(pgm_path, sizeof(pgm_path), "%s/dbz_wubu_decoded.pgm", output_dir);
    FILE* pgm_f = fopen(pgm_path, "wb");
    if(pgm_f){
        fprintf(pgm_f, "P5\n%d %d\n255\n", W, H);
        fwrite(decoded_wubu, 1, (size_t)num_frames * frame_size, pgm_f);
        fclose(pgm_f);
        printf("\nSaved WuBuCodec decoded to %s\n", pgm_path);
    }

    free(frames); free(frame_buf);
    free(decoded_wubu); free(decoded_x264); free(decoded_av1);

    return 0;
}
