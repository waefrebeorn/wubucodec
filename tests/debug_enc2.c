/* debug_enc2.c: Debug encoder loop */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"

#define W 320
#define H 240

int main() {
    printf("=== Debug Encoder Loop ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    uint8_t* prev = malloc(np);
    int first = 1;
    
    FILE* f = fopen("/tmp/balloon.raw", "rb");
    if(!f) { printf("Cannot open\n"); return 1; }
    
    int count = 0;
    while(fread(frame, 1, np, f) == (size_t)np) {
        long bits = 0;
        int frame_type = first ? WUBU_I_FRAME : WUBU_P_FRAME;
        int rc = wubu_encode_frame_preset(frame, first ? NULL : prev, NULL, W, H, 26, frame_type, NULL, recon, &bits, 2);
        
        if(rc != 0) {
            printf("Encode failed at frame %d: %d\n", count, rc);
            break;
        }
        
        memcpy(prev, recon, np);
        first = 0;
        count++;
        
        if(count % 100 == 0) {
            printf("  Frame %d: bits=%ld\n", count, bits);
        }
        
        if(count >= 100) {
            printf("  Stopping at 100 frames\n");
            break;
        }
    }
    
    fclose(f);
    free(frame);
    free(recon);
    free(prev);
    
    printf("Encoded %d frames\n", count);
    return 0;
}