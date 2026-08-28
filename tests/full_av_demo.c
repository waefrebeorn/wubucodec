/* full_av_demo.c: Complete A/V encoding + playback demo
 *
 * Generates test content, encodes with WuBuCodec RDO + Kodak audio,
 * saves to WUBV container, and plays back with SDL2 player.
 *
 * Usage: ./full_av_demo
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_encoder.h"
#include "wubuv_audio.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define W 640
#define H 480
#define FPS 30
#define N 90  /* 3 seconds */
#define AUDIO_RATE 48000
#define AUDIO_CH 1
#define AUDIO_SAMPLES_PER_FRAME (AUDIO_RATE / FPS)

/* Generate video frame with motion + detail */
static void generate_video_frame(uint8_t* frame, int f) {
    int cx = 320 + (int)(120 * cos(2.0 * M_PI * f / N));
    int cy = 240 + (int)(100 * sin(2.0 * M_PI * f / N));
    int edge = 200 + f * 3;
    int stripe = 80 + f * 2;
    
    for(int y = 0; y < H; y++) {
        for(int x = 0; x < W; x++) {
            double v = 128.0;
            
            /* Sharp moving edge */
            if(abs(x - edge) < 2) v = 255.0;
            else if(x < edge) v = 100.0;
            else v = 200.0;
            
            /* Fine texture */
            v += 15.0 * sin(2.0 * M_PI * 12.0 * x / W) * cos(2.0 * M_PI * 12.0 * y / H);
            
            /* Moving circle */
            int dx = x - cx, dy = y - cy;
            if(dx*dx + dy*dy < 1600) v = 255.0;
            
            /* Moving stripe */
            if(abs(y - stripe) < 5) v = 0.0;
            
            frame[y * W + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}

/* Generate audio frame (tones + sweep + noise) */
static void generate_audio_frame(float* samples, int num_samples, int frame_idx) {
    float base_freq = 440.0f + 220.0f * sin(2.0f * M_PI * frame_idx / N);
    
    for(int i = 0; i < num_samples; i++) {
        float t = (float)i / AUDIO_RATE;
        float global_t = (frame_idx * num_samples + i) / (float)AUDIO_RATE;
        
        /* Fundamental + harmonics */
        float s = 0.5f * sinf(2.0f * M_PI * base_freq * t)
                + 0.3f * sinf(2.0f * M_PI * base_freq * 2.0f * t + 1.0f)
                + 0.2f * sinf(2.0f * M_PI * base_freq * 3.0f * t + 2.0f);
        
        /* Frequency sweep */
        float sweep = 200.0f + 300.0f * global_t / (N / FPS);
        s += 0.3f * sinf(2.0f * M_PI * sweep * global_t);
        
        /* Noise */
        s += 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        
        samples[i] = s;
    }
}

int main(void) {
    printf("========================================\n");
    printf("  WuBuCodec Full A/V Demo\n");
    printf("========================================\n");
    
    const int np = W * H;
    uint8_t* video_frame = malloc(np);
    uint8_t* recon_frame = malloc(np);
    float* audio_frame = malloc(sizeof(float) * AUDIO_SAMPLES_PER_FRAME);
    
    /* ===== Encode Video Only (for comparison) ===== */
    printf("\n[1] Encoding video only (QP=26, fast preset)...\n");
    
    FILE* fvid = fopen("docs/video_examples/original_video.raw", "wb");
    FILE* fenc = fopen("docs/video_examples/encoded_qp26.raw", "wb");
    FILE* ferr = fopen("docs/video_examples/error_qp26.raw", "wb");
    FILE* fab = fopen("docs/video_examples/ab_comparison.raw", "wb");
    
    WubuvHeader hdr;
    wubuv_hdr_init(&hdr, W, H, FPS, N, 0, 0);
    WubuvWriter* writer = wubuv_writer_open("docs/video_examples/video_only.wubv", &hdr);
    
    double total_psnr = 0;
    int ab_w = W * 2;
    uint8_t* ab_buf = malloc(ab_w * H);
    
    for(int f = 0; f < N; f++) {
        /* Generate */
        generate_video_frame(video_frame, f);
        
        /* Save original */
        fwrite(video_frame, 1, np, fvid);
        
        /* Encode */
        long bits = 0;
        wubu_encode_frame_preset(video_frame, 
                                  f > 0 ? recon_frame : NULL,
                                  NULL, W, H, 26,
                                  f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME,
                                  NULL, recon_frame, &bits, 2);
        
        /* Save encoded */
        fwrite(recon_frame, 1, np, fenc);
        
        /* Write to WUBV */
        wubuv_write_frame(writer, recon_frame, f > 0);
        
        /* Error visualization */
        uint8_t* err_buf = malloc(np);
        for(int i = 0; i < np; i++) {
            int err = abs((int)video_frame[i] - (int)recon_frame[i]);
            err_buf[i] = (uint8_t)(err * 4 > 255 ? 255 : err * 4);
        }
        fwrite(err_buf, 1, np, ferr);
        free(err_buf);
        
        /* A/B comparison */
        memset(ab_buf, 0, ab_w * H);
        for(int y = 0; y < H; y++) {
            memcpy(ab_buf + y * ab_w, video_frame + y * W, W);
            memcpy(ab_buf + y * ab_w + W, recon_frame + y * W, W);
        }
        fwrite(ab_buf, 1, ab_w * H, fab);
        
        /* PSNR */
        double mse = 0;
        for(int i = 0; i < np; i++) {
            double d = (double)video_frame[i] - (double)recon_frame[i];
            mse += d * d;
        }
        mse /= np;
        double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        total_psnr += psnr;
    }
    
    wubuv_writer_close(writer);
    fclose(fvid);
    fclose(fenc);
    fclose(ferr);
    fclose(fab);
    free(ab_buf);
    
    printf("  Average PSNR: %.1f dB\n", total_psnr / N);
    printf("  WUBV file: video_only.wubv\n");
    
    /* ===== Encode Audio Only (Kodak) ===== */
    printf("\n[2] Encoding audio only (Kodak STFT)...\n");
    
    FILE* faud = fopen("docs/video_examples/original_audio.raw", "wb");
    FILE* faud_enc = fopen("docs/video_examples/encoded_audio.raw", "wb");
    
    WubuStft stft;
    WubuKodak kodak;
    wubu_stft_init(&stft, 512, 128);
    wubu_kodak_init(&kodak, 256, 256);
    
    int total_samples = N * AUDIO_SAMPLES_PER_FRAME;
    float* all_audio = malloc(sizeof(float) * total_samples);
    float* all_encoded = malloc(sizeof(float) * total_samples);
    
    /* Generate all audio */
    for(int f = 0; f < N; f++) {
        generate_audio_frame(audio_frame, AUDIO_SAMPLES_PER_FRAME, f);
        memcpy(all_audio + f * AUDIO_SAMPLES_PER_FRAME, audio_frame, sizeof(float) * AUDIO_SAMPLES_PER_FRAME);
        fwrite(audio_frame, sizeof(float), AUDIO_SAMPLES_PER_FRAME, faud);
    }
    
    /* Encode all audio via Kodak */
    int frames;
    float* S = wubu_stft_forward(&stft, all_audio, total_samples, &frames);
    wubu_kodak_pack(&kodak, S, frames, stft.frame_size / 2 + 1);
    
    /* Decode */
    float* S2 = malloc(sizeof(float) * frames * (stft.frame_size / 2 + 1) * 2);
    wubu_kodak_unpack(&kodak, S2, frames, stft.frame_size / 2 + 1);
    float* decoded = wubu_stft_inverse(&stft, S2, frames, total_samples);
    
    /* Save encoded audio */
    fwrite(decoded, sizeof(float), total_samples, faud_enc);
    
    /* Compute audio correlation */
    double corr = 0, dx = 0, dy = 0;
    for(int i = 1000; i < total_samples - 1000; i++) {
        corr += all_audio[i] * decoded[i];
        dx += all_audio[i] * all_audio[i];
        dy += decoded[i] * decoded[i];
    }
    float audio_corr = (float)(corr / (sqrt(dx * dy) + 1e-12));
    
    printf("  Audio round-trip correlation: %.4f\n", audio_corr);
    printf("  STFT frames: %d, bins: %d\n", frames, stft.frame_size / 2 + 1);
    
    fclose(faud);
    fclose(faud_enc);
    free(all_audio);
    free(all_encoded);
    free(S);
    free(S2);
    free(decoded);
    wubu_kodak_free(&kodak);
    wubu_stft_free(&stft);
    
    /* ===== Full A/V encode in WUBV container ===== */
    printf("\n[3] Full A/V encode (WUBV container with audio)...\n");
    
    WubuvHeader hdr_av;
    wubuv_hdr_init(&hdr_av, W, H, FPS, N, 1, AUDIO_RATE);
    WubuvAudioWriter* av_writer = wubuv_audio_writer_open(
        "docs/video_examples/full_av.wubv", W, H, FPS, N, 1, AUDIO_RATE, AUDIO_CH);
    
    if(!av_writer) {
        printf("  Failed to create A/V writer\n");
        return 1;
    }
    
    for(int f = 0; f < N; f++) {
        /* Generate video */
        generate_video_frame(video_frame, f);
        
        /* Generate audio */
        generate_audio_frame(audio_frame, AUDIO_SAMPLES_PER_FRAME, f);
        
        /* Write to container */
        wubuv_audio_write_video(av_writer, video_frame, f > 0);
        wubuv_audio_write_audio(av_writer, audio_frame, AUDIO_SAMPLES_PER_FRAME);
    }
    
    wubuv_audio_writer_close(av_writer);
    printf("  Created: full_av.wubv\n");
    
    /* ===== Verify the file ===== */
    printf("\n[4] Verifying WUBV file...\n");
    WubuvAudioReader* reader = wubuv_audio_reader_open("docs/video_examples/full_av.wubv");
    if(reader) {
        printf("  Video: %dx%d @ %d fps\n", reader->base.hdr.width, reader->base.hdr.height, reader->base.hdr.fps);
        printf("  Audio: %d Hz, %d channels\n", reader->base.hdr.audio_rate, reader->has_audio);
        printf("  Frames: %d\n", reader->base.hdr.frame_count);
        
        /* Read some frames to verify */
        uint8_t* read_frame = malloc(np);
        int read_frames = 0;
        int is_inter;
        while(wubuv_audio_read_video(reader, read_frame, &is_inter) == 0) {
            read_frames++;
        }
        printf("  Read back %d video frames\n", read_frames);
        free(read_frame);
        wubuv_audio_reader_close(reader);
    }
    
    /* ===== Also create video at different QPs ===== */
    printf("\n[5] Encoding at different QPs...\n");
    int qps[] = {20, 26, 32, 40};
    for(int q = 0; q < 4; q++) {
        char fname[256];
        snprintf(fname, sizeof(fname), "docs/video_examples/encoded_qp%d.raw", qps[q]);
        FILE* f = fopen(fname, "wb");
        
        uint8_t* recon = malloc(np);
        double psnr = 0;
        for(int f_idx = 0; f_idx < N; f_idx++) {
            generate_video_frame(video_frame, f_idx);
            long bits = 0;
            wubu_encode_frame_preset(video_frame,
                                      f_idx > 0 ? recon : NULL,
                                      NULL, W, H, qps[q],
                                      f_idx == 0 ? WUBU_I_FRAME : WUBU_P_FRAME,
                                      NULL, recon, &bits, 2);
            fwrite(recon, 1, np, f);
            
            double mse = 0;
            for(int i = 0; i < np; i++) {
                double d = (double)video_frame[i] - (double)recon[i];
                mse += d * d;
            }
            mse /= np;
            psnr += mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        }
        psnr /= N;
        fclose(f);
        free(recon);
        printf("  QP=%d: %.1f dB PSNR\n", qps[q], psnr);
    }
    
    /* Cleanup */
    free(video_frame);
    free(recon_frame);
    free(audio_frame);
    
    printf("\n========================================\n");
    printf("  Done! Files created:\n");
    printf("========================================\n");
    printf("  original_video.raw      - Test content\n");
    printf("  encoded_qp26.raw        - Encoded QP=26\n");
    printf("  error_qp26.raw          - Error visualization\n");
    printf("  ab_comparison.raw       - Side-by-side A/B\n");
    printf("  video_only.wubv         - WUBV video container\n");
    printf("  full_av.wubv            - WUBV with audio\n");
    printf("\nConvert to MP4 with:\n");
    printf("  ffmpeg -f rawvideo -pix_fmt gray -s 640x480 -r 30 -i original_video.raw -c:v libx264 original_video.mp4\n");
    printf("  ffmpeg -f rawvideo -pix_fmt gray -s 1280x480 -r 30 -i ab_comparison.raw -c:v libx264 ab_comparison.mp4\n");
    
    return 0;
}
