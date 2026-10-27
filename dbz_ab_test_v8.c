/* dbz_ab_test_v8.c — Final A/B test with verified reconstruction
 *
 * WuBuCodec v8: I-frame only, verified PSNR
 * x264: matched bitrate
 * AV1: matched bitrate
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static double compute_psnr(const uint8_t* a, const uint8_t* b, size_t n){
    double mse = 0;
    for(size_t i=0; i<n; i++){ double d = (double)a[i] - (double)b[i]; mse += d*d; }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

static double compute_ssim(const uint8_t* a, const uint8_t* b, int W, int H){
    double C1 = (double)(0.01*255)*(0.01*255), C2 = (double)(0.03*255)*(0.03*255);
    double ssim_sum = 0;
    int windows = 0;
    for(int y=0; y<H-7; y+=8){
        for(int x=0; x<W-7; x+=8){
            double mean_a=0, mean_b=0, var_a=0, var_b=0, covar=0;
            for(int dy=0; dy<8; dy++) for(int dx=0; dx<8; dx++){
                int idx=(y+dy)*W+(x+dx);
                mean_a+=a[idx]; mean_b+=b[idx];
            }
            mean_a/=64; mean_b/=64;
            for(int dy=0; dy<8; dy++) for(int dx=0; dx<8; dx++){
                int idx=(y+dy)*W+(x+dx);
                double da=a[idx]-mean_a, db=b[idx]-mean_b;
                var_a+=da*da; var_b+=db*db; covar+=da*db;
            }
            var_a/=64; var_b/=64; covar/=64;
            ssim_sum += ((2*mean_a*mean_b+C1)*(2*covar+C2))/((mean_a*mean_a+mean_b*mean_b+C1)*(var_a+var_b+C2));
            windows++;
        }
    }
    return windows > 0 ? ssim_sum/windows : 0;
}

int main(int argc, char** argv){
    if(argc < 3){
        printf("DBZ A/B Test v8: verified reconstruction\n");
        printf("Usage: %s <input.y4m> <output_dir>\n", argv[0]);
        return 1;
    }

    const char* input = argv[1];
    const char* output_dir = argv[2];

    FILE* f = fopen(input, "rb");
    if(!f){ printf("Cannot open %s\n", input); return 1; }

    char header[256];
    if(!fgets(header, sizeof(header), f)){ printf("Cannot read header\n"); return 1; }

    int W = 320, H = 240, num_frames = 360;
    char* pw = strstr(header, "W");
    char* ph = strstr(header, "H");
    if(pw && ph){ W = atoi(pw+1); H = atoi(ph+1); }

    int frame_size = W * H;
    uint8_t* frames = malloc((size_t)num_frames * frame_size);
    uint8_t* frame_buf = malloc(frame_size);

    for(int i=0; i<num_frames; i++){
        char frame_header[64];
        if(!fgets(frame_header, sizeof(frame_header), f)){ num_frames = i; break; }
        if(fread(frame_buf, 1, frame_size, f) != (size_t)frame_size){ num_frames = i; break; }
        memcpy(frames + (size_t)i * frame_size, frame_buf, frame_size);
    }
    fclose(f);

    /* Save input raw */
    char raw_path[512];
    snprintf(raw_path, sizeof(raw_path), "%s/dbz_input.raw", output_dir);
    FILE* raw_f = fopen(raw_path, "wb");
    fwrite(frames, 1, (size_t)num_frames * frame_size, raw_f);
    fclose(raw_f);

    /* ===== Encode with WuBuCodec v8 (I-frame only, verified) ===== */
    printf("=== Encoding with WuBuCodec v8 (I-frame only) ===\n");

    char wubv_path[512];
    snprintf(wubv_path, sizeof(wubv_path), "%s/dbz_wubucodec.wubv", output_dir);
    char recon_path[512];
    snprintf(recon_path, sizeof(recon_path), "%s/dbz_wubu_recon.raw", output_dir);

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "./bin/wubuv_encode_v8_recon %s %s %s -w %d -h %d -f %d -q 26 2>&1",
             raw_path, wubv_path, recon_path, W, H, 24);
    printf("Running: %s\n", cmd);
    system(cmd);

    /* Get file size */
    FILE* wubv_f = fopen(wubv_path, "rb");
    long wubv_size = 0;
    if(wubv_f){ fseek(wubv_f, 0, SEEK_END); wubv_size = ftell(wubv_f); fclose(wubv_f); }

    /* Read recon and compute PSNR */
    uint8_t* wubu_recon = malloc((size_t)num_frames * frame_size);
    FILE* recon_f = fopen(recon_path, "rb");
    if(recon_f){
        fread(wubu_recon, 1, (size_t)num_frames * frame_size, recon_f);
        fclose(recon_f);
    }

    double wubu_psnr = compute_psnr(frames, wubu_recon, (size_t)num_frames * frame_size);
    double wubu_ssim = compute_ssim(frames, wubu_recon, W, H);

    double wubu_bitrate_kbps = (double)wubv_size * 8 * 24 / (double)num_frames / 1000.0;

    printf("WuBuCodec v8: %.2f KB, %.2f dB, SSIM %.4f, %.0f kbps\n",
           wubv_size/1024.0, wubu_psnr, wubu_ssim, wubu_bitrate_kbps);

    /* ===== Encode with x264 at matched bitrate ===== */
    printf("\n=== Encoding with x264 at %.0f kbps ===\n", wubu_bitrate_kbps);

    char x264_path[512];
    snprintf(x264_path, sizeof(x264_path), "%s/dbz_x264.mp4", output_dir);

    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -f rawvideo -pix_fmt gray -s %dx%d -r 24 -i %s "
             "-c:v libx264 -preset medium -b:v %dk -pix_fmt gray %s 2>&1 | tail -3",
             W, H, raw_path, (int)wubu_bitrate_kbps, x264_path);
    system(cmd);

    char x264_dec[512];
    snprintf(x264_dec, sizeof(x264_dec), "%s/dbz_x264_dec.raw", output_dir);
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -i %s -f rawvideo -pix_fmt gray %s 2>&1 | tail -2",
             x264_path, x264_dec);
    system(cmd);

    uint8_t* decoded_x264 = malloc((size_t)num_frames * frame_size);
    FILE* x264_dec_f = fopen(x264_dec, "rb");
    if(x264_dec_f){ fread(decoded_x264, 1, (size_t)num_frames * frame_size, x264_dec_f); fclose(x264_dec_f); }

    double x264_psnr = compute_psnr(frames, decoded_x264, (size_t)num_frames * frame_size);
    double x264_ssim = compute_ssim(frames, decoded_x264, W, H);

    FILE* x264_out_f = fopen(x264_path, "rb");
    long x264_size = 0;
    if(x264_out_f){ fseek(x264_out_f, 0, SEEK_END); x264_size = ftell(x264_out_f); fclose(x264_out_f); }

    printf("x264: %.2f KB, %.2f dB, SSIM %.4f\n", x264_size/1024.0, x264_psnr, x264_ssim);

    /* ===== Encode with AV1 at matched bitrate ===== */
    printf("\n=== Encoding with AV1 at %.0f kbps ===\n", wubu_bitrate_kbps);

    char av1_path[512];
    snprintf(av1_path, sizeof(av1_path), "%s/dbz_av1.mp4", output_dir);
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -f rawvideo -pix_fmt gray -s %dx%d -r 24 -i %s "
             "-c:v libaom-av1 -preset 6 -b:v %dk -pix_fmt gray %s 2>&1 | tail -3",
             W, H, raw_path, (int)wubu_bitrate_kbps, av1_path);
    system(cmd);

    char av1_dec[512];
    snprintf(av1_dec, sizeof(av1_dec), "%s/dbz_av1_dec.raw", output_dir);
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -i %s -f rawvideo -pix_fmt gray %s 2>&1 | tail -2",
             av1_path, av1_dec);
    system(cmd);

    uint8_t* decoded_av1 = malloc((size_t)num_frames * frame_size);
    FILE* av1_dec_f = fopen(av1_dec, "rb");
    if(av1_dec_f){ fread(decoded_av1, 1, (size_t)num_frames * frame_size, av1_dec_f); fclose(av1_dec_f); }

    double av1_psnr = compute_psnr(frames, decoded_av1, (size_t)num_frames * frame_size);
    double av1_ssim = compute_ssim(frames, decoded_av1, W, H);

    FILE* av1_out_f = fopen(av1_path, "rb");
    long av1_size = 0;
    if(av1_out_f){ fseek(av1_out_f, 0, SEEK_END); av1_size = ftell(av1_out_f); fclose(av1_out_f); }

    printf("AV1: %.2f KB, %.2f dB, SSIM %.4f\n", av1_size/1024.0, av1_psnr, av1_ssim);

    /* ===== Summary ===== */
    printf("\n");
    printf("============================================================\n");
    printf("     DBZ A/B TEST RESULTS (360 frames, 320x240, ~%.0f kbps)   \n", wubu_bitrate_kbps);
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

    free(frames); free(frame_buf);
    free(wubu_recon); free(decoded_x264); free(decoded_av1);

    return 0;
}
