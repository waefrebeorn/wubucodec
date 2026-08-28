#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_encoder.h"

int main() {
    printf("=== WUBV Direct Test ===\n");
    
    // Use the encoder directly, bypass WUBV container
    int W = 320, H = 240;
    int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    
    // Generate test pattern
    for(int i = 0; i < np; i++) {
        int x = i % W;
        int y = i / W;
        frame[i] = (uint8_t)((x + y) % 256);
    }
    
    // Encode
    long bits = 0;
    int rc = wubu_encode_frame(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits);
    printf("Encode: rc=%d, bits=%ld\n", rc, bits);
    
    // Save raw frames for ffmpeg
    FILE* f = fopen("docs/video_examples/direct_encoded.raw", "wb");
    fwrite(recon, 1, np, f);
    fclose(f);
    
    printf("Saved direct_encoded.raw\n");
    
    free(frame);
    free(recon);
    return 0;
}
