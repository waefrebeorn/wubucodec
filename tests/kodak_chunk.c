/* kodak_chunk.c: Kodak audio in chunks for long audio */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_kodak.h"
#include "wubu_stft.h"

#define AUDIO_RATE 48000
#define KODAK_W 256
#define KODAK_H 256
#define KODAK_PIXELS (KODAK_W * KODAK_H)

int main() {
    printf("=== Kodak Audio (Chunked) ===\n");
    
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
    
    /* STFT setup */
    WubuStft stft;
    wubu_stft_init(&stft, 512, 128);
    int bins = stft.frame_size / 2 + 1;  /* 257 */
    
    /* Max frames that fit in Kodak image */
    int max_frames_per_chunk = KODAK_PIXELS / bins;  /* 255 */
    int max_samples_per_chunk = (max_frames_per_chunk - 1) * stft.hop + stft.frame_size;
    
    printf("Kodak: %dx%d = %d pixels, %d bins, max %d frames/chunk, %.2f sec/chunk\n",
           KODAK_W, KODAK_H, KODAK_PIXELS, bins, max_frames_per_chunk, 
           (double)max_samples_per_chunk / AUDIO_RATE);
    
    /* Process in chunks */
    float* all_decoded = malloc(sizeof(float) * total_samples);
    int total_corr_samples = 0;
    double total_corr = 0, total_dx = 0, total_dy = 0;
    int chunk_num = 0;
    int offset = 0;
    
    while(offset < total_samples) {
        int chunk_samples = total_samples - offset;
        if(chunk_samples > max_samples_per_chunk) chunk_samples = max_samples_per_chunk;
        
        /* Forward STFT */
        int frames;
        float* S = wubu_stft_forward(&stft, audio + offset, chunk_samples, &frames);
        
        /* Pack into Kodak */
        WubuKodak kodak;
        wubu_kodak_init(&kodak, KODAK_W, KODAK_H);
        wubu_kodak_pack(&kodak, S, frames, bins);
        
        /* Unpack from Kodak */
        float* S2 = malloc(sizeof(float) * frames * bins * 2);
        wubu_kodak_unpack(&kodak, S2, frames, bins);
        
        /* Inverse STFT */
        float* decoded = wubu_stft_inverse(&stft, S2, frames, chunk_samples);
        
        /* Copy to output */
        memcpy(all_decoded + offset, decoded, sizeof(float) * chunk_samples);
        
        /* Correlation for this chunk */
        for(int i = 500; i < chunk_samples - 500; i++) {
            total_corr += audio[offset + i] * decoded[i];
            total_dx += audio[offset + i] * audio[offset + i];
            total_dy += decoded[i] * decoded[i];
            total_corr_samples++;
        }
        
        free(S);
        free(S2);
        free(decoded);
        wubu_kodak_free(&kodak);
        
        offset += chunk_samples;
        chunk_num++;
        
        if(chunk_num % 10 == 1) {
            printf("  Chunk %d: offset=%d, samples=%d, frames=%d\n", chunk_num, offset, chunk_samples, frames);
        }
    }
    
    float audio_corr = (float)(total_corr / (sqrt(total_dx * total_dy) + 1e-12));
    printf("Total chunks: %d, samples: %d\n", chunk_num, total_corr_samples);
    printf("Audio round-trip correlation: %.4f\n", audio_corr);
    
    /* Save */
    FILE* fout = fopen("/tmp/balloon_kodak.raw", "wb");
    fwrite(all_decoded, sizeof(float), total_samples, fout);
    fclose(fout);
    
    printf("Saved balloon_kodak.raw\n");
    
    free(audio);
    free(all_decoded);
    wubu_stft_free(&stft);
    
    return 0;
}