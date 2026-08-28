/* kodak_audio.c: Encode/decode audio with Kodak STFT */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_kodak.h"
#include "wubu_stft.h"

#define AUDIO_RATE 48000

int main() {
    printf("=== Kodak Audio Test ===\n");
    
    /* Read audio */
    FILE* f = fopen("/tmp/balloon_audio.raw", "rb");
    if(!f) { printf("Cannot open audio\n"); return 1; }
    
    fseek(f, 0, SEEK_END);
    long audio_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    int total_samples = audio_size / 4;
    float* audio = malloc(audio_size);
    fread(audio, 1, audio_size, f);
    fclose(f);
    
    printf("Audio: %d samples, %.1f seconds\n", total_samples, (double)total_samples / AUDIO_RATE);
    
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
    double corr = 0, dx = 0, dy = 0;
    for(int i = 1000; i < total_samples - 1000; i++) {
        corr += audio[i] * decoded[i];
        dx += audio[i] * audio[i];
        dy += decoded[i] * decoded[i];
    }
    float audio_corr = (float)(corr / (sqrt(dx * dy) + 1e-12));
    printf("Audio round-trip correlation: %.4f\n", audio_corr);
    
    /* Save */
    FILE* fout = fopen("/tmp/balloon_kodak.raw", "wb");
    fwrite(decoded, sizeof(float), total_samples, fout);
    fclose(fout);
    
    printf("Saved balloon_kodak.raw\n");
    
    free(audio);
    free(S);
    free(S2);
    free(decoded);
    wubu_kodak_free(&kodak);
    wubu_stft_free(&stft);
    
    return 0;
}