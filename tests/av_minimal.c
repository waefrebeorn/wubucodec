/* av_minimal.c: Minimal working A/V demo */
#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_encoder.h"
#include "wubu_kodak.h"
#include "wubu_stft.h"

#define W 320
#define H 240
#define FPS 30
#define N 30
#define AUDIO_RATE 48000
#define AUDIO_SAMPLES (AUDIO_RATE / FPS)

int main(void) {
    printf("=== Minimal A/V Demo ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    float* audio = malloc(sizeof(float) * AUDIO_SAMPLES);
    
    /* Step 1: Generate and encode video */
    printf("Encoding %d video frames at %dx%d...\n", N, W, H);
    
    FILE* fvid = fopen("docs/video_examples/mini_video.raw", "wb");
    FILE* fenc = fopen("docs/video_examples/mini_encoded.raw", "wb");
    
    uint8_t* prev = NULL;
    double total_psnr = 0;
    
    for(int f_idx = 0; f_idx < N; f_idx++) {
        /* Generate frame */
        for(int i = 0; i < np; i++) {
            int x = i % W;
            int y = i / W;
            frame[i] = (uint8_t)((x + y + f_idx * 3) % 256);
        }
        fwrite(frame, 1, np, fvid);
        
        /* Encode */
        long bits = 0;
        int frame_type = (f_idx == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        uint8_t* ref = (f_idx > 0) ? prev : NULL;
        
        int rc = wubu_encode_frame_preset(frame, ref, NULL, W, H, 26, frame_type, NULL, recon, &bits, 2);
        if(rc != 0) {
            printf("Encode failed at frame %d: %d\n", f_idx, rc);
            break;
        }
        
        fwrite(recon, 1, np, fenc);
        
        if(!prev) prev = malloc(np);
        memcpy(prev, recon, np);
        
        /* PSNR */
        double mse = 0;
        for(int i = 0; i < np; i++) {
            double d = (double)frame[i] - (double)recon[i];
            mse += d * d;
        }
        mse /= np;
        double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
        total_psnr += psnr;
    }
    
    fclose(fvid);
    fclose(fenc);
    free(prev);
    
    printf("  Average PSNR: %.1f dB\n", total_psnr / N);
    
    /* Step 2: Generate and encode audio with Kodak */
    printf("Encoding %d audio frames at %d Hz...\n", N, AUDIO_RATE);
    
    FILE* faud = fopen("docs/video_examples/mini_audio.raw", "wb");
    FILE* faud_enc = fopen("docs/video_examples/mini_audio_enc.raw", "wb");
    
    WubuStft stft;
    WubuKodak kodak;
    wubu_stft_init(&stft, 512, 128);
    wubu_kodak_init(&kodak, 256, 256);
    
    /* Generate all audio */
    int total_samples = N * AUDIO_SAMPLES;
    float* all_audio = malloc(sizeof(float) * total_samples);
    
    for(int f_idx = 0; f_idx < N; f_idx++) {
        /* Generate audio (simple tone + sweep) */
        float base_freq = 440.0f + 220.0f * sinf(2.0f * M_PI * f_idx / N);
        for(int i = 0; i < AUDIO_SAMPLES; i++) {
            float t = (float)i / AUDIO_RATE;
            float s = 0.5f * sinf(2.0f * M_PI * base_freq * t)
                    + 0.3f * sinf(2.0f * M_PI * base_freq * 2.0f * t);
            all_audio[f_idx * AUDIO_SAMPLES + i] = s;
        }
        fwrite(all_audio + f_idx * AUDIO_SAMPLES, sizeof(float), AUDIO_SAMPLES, faud);
    }
    
    /* Encode with Kodak */
    int frames;
    float* S = wubu_stft_forward(&stft, all_audio, total_samples, &frames);
    wubu_kodak_pack(&kodak, S, frames, stft.frame_size / 2 + 1);
    
    /* Decode */
    float* S2 = malloc(sizeof(float) * frames * (stft.frame_size / 2 + 1) * 2);
    wubu_kodak_unpack(&kodak, S2, frames, stft.frame_size / 2 + 1);
    float* decoded = wubu_stft_inverse(&stft, S2, frames, total_samples);
    
    fwrite(decoded, sizeof(float), total_samples, faud_enc);
    
    /* Correlation */
    double corr = 0, dx = 0, dy = 0;
    for(int i = 1000; i < total_samples - 1000; i++) {
        corr += all_audio[i] * decoded[i];
        dx += all_audio[i] * all_audio[i];
        dy += decoded[i] * decoded[i];
    }
    float audio_corr = (float)(corr / (sqrt(dx * dy) + 1e-12));
    printf("  Audio round-trip correlation: %.4f\n", audio_corr);
    
    fclose(faud);
    fclose(faud_enc);
    free(all_audio);
    free(S);
    free(S2);
    free(decoded);
    wubu_kodak_free(&kodak);
    wubu_stft_free(&stft);
    
    /* Step 3: Create combined A/V file */
    printf("Creating combined A/V...\n");
    
    /* Create a simple interleaved file: video frame + audio frame */
    FILE* fav = fopen("docs/video_examples/mini_av.raw", "wb");
    
    /* Re-encode video and interleave with audio */
    prev = NULL;
    for(int f_idx = 0; f_idx < N; f_idx++) {
        /* Generate video */
        for(int i = 0; i < np; i++) {
            int x = i % W;
            int y = i / W;
            frame[i] = (uint8_t)((x + y + f_idx * 3) % 256);
        }
        
        /* Encode video */
        long bits = 0;
        int frame_type = (f_idx == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        uint8_t* ref = (f_idx > 0) ? prev : NULL;
        wubu_encode_frame_preset(frame, ref, NULL, W, H, 26, frame_type, NULL, recon, &bits, 2);
        
        if(!prev) prev = malloc(np);
        memcpy(prev, recon, np);
        
        /* Write video frame */
        fwrite(recon, 1, np, fav);
        
        /* Generate and write audio frame */
        float base_freq = 440.0f + 220.0f * sinf(2.0f * M_PI * f_idx / N);
        for(int i = 0; i < AUDIO_SAMPLES; i++) {
            float t = (float)i / AUDIO_RATE;
            audio[i] = 0.5f * sinf(2.0f * M_PI * base_freq * t)
                     + 0.3f * sinf(2.0f * M_PI * base_freq * 2.0f * t);
        }
        fwrite(audio, sizeof(float), AUDIO_SAMPLES, fav);
    }
    
    fclose(fav);
    free(prev);
    
    free(frame);
    free(recon);
    free(audio);
    
    printf("\n=== Done! ===\n");
    printf("Files created:\n");
    printf("  mini_video.raw      - Original video\n");
    printf("  mini_encoded.raw    - WuBuCodec encoded\n");
    printf("  mini_audio.raw      - Original audio\n");
    printf("  mini_audio_enc.raw  - Kodak encoded audio\n");
    printf("  mini_av.raw         - Interleaved A/V\n");
    printf("\nConvert with ffmpeg:\n");
    printf("  ffmpeg -f rawvideo -pix_fmt gray -s 320x240 -r 30 -i mini_video.raw -c:v libx264 mini_video.mp4\n");
    printf("  ffmpeg -f f32le -ar 48000 -ac 1 -i mini_audio.raw mini_audio.wav\n");
    
    return 0;
}