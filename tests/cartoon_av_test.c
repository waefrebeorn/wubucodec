/* cartoon_av_test.c: Full A/V test with real cartoon from Internet Archive */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_encoder.h"
#include "wubu_kodak.h"
#include "wubu_stft.h"
#include "wubuv.h"

#define W 320
#define H 240
#define FPS 25
#define AUDIO_RATE 48000

/* Compute PSNR */
static double compute_psnr(const uint8_t* a, const uint8_t* b, int n) {
    double mse = 0;
    for(int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
}

/* Compute correlation */
static double compute_corr(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for(int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrt(na * nb) + 1e-12);
}

int main(int argc, char** argv) {
    if(argc < 3) {
        printf("Usage: %s video.raw audio.raw [num_frames]\n", argv[0]);
        return 1;
    }
    
    const char* videofile = argv[1];
    const char* audiofile = argv[2];
    int num_frames = (argc > 3) ? atoi(argv[3]) : 3675;
    
    printf("========================================\n");
    printf("  WuBuCodec Real Cartoon A/V Test\n");
    printf("========================================\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    uint8_t* prev = NULL;
    
    /* ===== Video Encoding ===== */
    printf("\n[1] Encoding %d video frames at %dx%d...\n", num_frames, W, H);
    
    FILE* fvid = fopen(videofile, "rb");
    FILE* fenc = fopen("/tmp/balloon_enc.raw", "wb");
    FILE* ferr = fopen("/tmp/balloon_err.raw", "wb");
    FILE* fab = fopen("/tmp/balloon_ab.raw", "wb");
    
    if(!fvid) { printf("Cannot open %s\n", videofile); return 1; }
    
    double total_psnr = 0;
    int ab_w = W * 2;
    uint8_t* ab_buf = malloc(ab_w * H);
    uint8_t* err_buf = malloc(np);
    int encoded_frames = 0;
    
    for(int f = 0; f < num_frames; f++) {
        if(fread(frame, 1, np, fvid) != (size_t)np) {
            printf("  Short read at frame %d\n", f);
            break;
        }
        
        /* Encode */
        long bits = 0;
        int frame_type = (f == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        uint8_t* ref = (f > 0) ? prev : NULL;
        
        int rc = wubu_encode_frame_preset(frame, ref, NULL, W, H, 26, frame_type, NULL, recon, &bits, 2);
        if(rc != 0) {
            printf("  Encode failed at frame %d: %d\n", f, rc);
            break;
        }
        
        fwrite(recon, 1, np, fenc);
        
        /* PSNR */
        double p = compute_psnr(frame, recon, np);
        total_psnr += p;
        
        /* Error visualization */
        for(int i = 0; i < np; i++) {
            int err = abs((int)frame[i] - (int)recon[i]);
            err_buf[i] = (uint8_t)(err * 4 > 255 ? 255 : err * 4);
        }
        fwrite(err_buf, 1, np, ferr);
        
        /* A/B comparison */
        memset(ab_buf, 0, ab_w * H);
        for(int y = 0; y < H; y++) {
            memcpy(ab_buf + y * ab_w, frame + y * W, W);
            memcpy(ab_buf + y * ab_w + W, recon + y * W, W);
        }
        fwrite(ab_buf, 1, ab_w * H, fab);
        
        if(!prev) prev = malloc(np);
        memcpy(prev, recon, np);
        
        encoded_frames++;
        
        if(f % 500 == 0) {
            printf("  Frame %d: PSNR=%.1f dB\n", f, p);
        }
    }
    
    fclose(fvid);
    fclose(fenc);
    fclose(ferr);
    fclose(fab);
    free(ab_buf);
    free(err_buf);
    free(prev);
    
    printf("  Encoded %d frames, Average PSNR: %.1f dB\n", encoded_frames, total_psnr / encoded_frames);
    
    /* ===== Audio Encoding (Kodak) ===== */
    printf("\n[2] Encoding audio with Kodak STFT...\n");
    
    FILE* faud = fopen(audiofile, "rb");
    if(!faud) { printf("Cannot open %s\n", audiofile); return 1; }
    
    /* Get file size */
    fseek(faud, 0, SEEK_END);
    long audio_size = ftell(faud);
    fseek(faud, 0, SEEK_SET);
    
    int total_samples = audio_size / 4;
    float* audio = malloc(audio_size);
    fread(audio, 1, audio_size, faud);
    fclose(faud);
    
    printf("  Audio: %d samples, %.1f seconds\n", total_samples, (double)total_samples / AUDIO_RATE);
    
    /* Kodak encode */
    WubuStft stft;
    WubuKodak kodak;
    wubu_stft_init(&stft, 512, 128);
    wubu_kodak_init(&kodak, 256, 256);
    
    int frames;
    float* S = wubu_stft_forward(&stft, audio, total_samples, &frames);
    wubu_kodak_pack(&kodak, S, frames, stft.frame_size / 2 + 1);
    
    /* Kodak decode */
    float* S2 = malloc(sizeof(float) * frames * (stft.frame_size / 2 + 1) * 2);
    wubu_kodak_unpack(&kodak, S2, frames, stft.frame_size / 2 + 1);
    float* decoded = wubu_stft_inverse(&stft, S2, frames, total_samples);
    
    /* Correlation */
    double corr = compute_corr(audio, decoded, total_samples);
    printf("  Audio round-trip correlation: %.4f\n", corr);
    
    /* Save encoded audio */
    FILE* faud_enc = fopen("/tmp/balloon_audio_enc.raw", "wb");
    fwrite(decoded, sizeof(float), total_samples, faud_enc);
    fclose(faud_enc);
    
    /* Save original audio for comparison */
    FILE* faud_orig = fopen("/tmp/balloon_audio_orig.raw", "wb");
    fwrite(audio, sizeof(float), total_samples, faud_orig);
    fclose(faud_orig);
    
    free(audio);
    free(S);
    free(S2);
    free(decoded);
    wubu_kodak_free(&kodak);
    wubu_stft_free(&stft);
    
    /* ===== Create WUBV Container ===== */
    printf("\n[3] Creating WUBV container...\n");
    
    /* Re-encode video and save to WUBV */
    FILE* fvid2 = fopen(videofile, "rb");
    if(!fvid2) { printf("Cannot open %s\n", videofile); return 1; }
    
    WubuvHeader hdr;
    wubuv_hdr_init(&hdr, W, H, FPS, encoded_frames, 0, 0);
    WubuvWriter* writer = wubuv_writer_open("/tmp/balloon.wubv", &hdr);
    
    prev = NULL;
    for(int f = 0; f < encoded_frames; f++) {
        if(fread(frame, 1, np, fvid2) != (size_t)np) break;
        
        long bits = 0;
        int frame_type = (f == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        uint8_t* ref = (f > 0) ? prev : NULL;
        
        wubu_encode_frame_preset(frame, ref, NULL, W, H, 26, frame_type, NULL, recon, &bits, 2);
        wubuv_write_frame(writer, recon, f > 0);
        
        if(!prev) prev = malloc(np);
        memcpy(prev, recon, np);
    }
    
    wubuv_writer_close(writer);
    fclose(fvid2);
    free(prev);
    
    printf("  Created balloon.wubv\n");
    
    /* ===== Verify WUBV ===== */
    printf("\n[4] Verifying WUBV...\n");
    WubuvReader* reader = wubuv_reader_open("/tmp/balloon.wubv");
    if(reader) {
        printf("  WUBV: %dx%d @ %d fps, %d frames\n", 
               reader->hdr.width, reader->hdr.height, reader->hdr.fps, reader->hdr.frame_count);
        
        uint8_t* read_frame = malloc(np);
        int count = 0;
        int is_inter;
        while(wubuv_read_frame(reader, read_frame, &is_inter) == 0) {
            count++;
        }
        printf("  Read back %d frames\n", count);
        free(read_frame);
        wubuv_reader_close(reader);
    }
    
    free(frame);
    free(recon);
    
    printf("\n========================================\n");
    printf("  Done! Files created:\n");
    printf("========================================\n");
    printf("  /tmp/balloon_enc.raw     - WuBuCodec encoded video\n");
    printf("  /tmp/balloon_err.raw     - Error visualization\n");
    printf("  /tmp/balloon_ab.raw      - A/B comparison\n");
    printf("  /tmp/balloon_audio_orig.raw  - Original audio\n");
    printf("  /tmp/balloon_audio_enc.raw   - Kodak encoded audio\n");
    printf("  /tmp/balloon.wubv        - WUBV container\n");
    printf("\nConvert to MP4 with:\n");
    printf("  ffmpeg -f rawvideo -pix_fmt gray -s 320x240 -r 25 -i balloon_enc.raw -c:v libx264 balloon_enc.mp4\n");
    printf("  ffmpeg -f rawvideo -pix_fmt gray -s 640x240 -r 25 -i balloon_ab.raw -c:v libx264 balloon_ab.mp4\n");
    
    return 0;
}