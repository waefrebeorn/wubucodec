#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_encoder.h"

int main() {
    printf("=== Encoder Debug ===\n");
    
    int W = 8, H = 4;
    int np = W * H;
    uint8_t frame[32];
    uint8_t recon[32];
    
    for(int i = 0; i < np; i++) frame[i] = (uint8_t)(i * 7);
    
    printf("Frame: ");
    for(int i = 0; i < np; i++) printf("%d ", frame[i]);
    printf("\n");
    
    long bits = 0;
    printf("Calling wubu_encode_frame(WUBU_I_FRAME)...\n");
    fflush(stdout);
    int rc = wubu_encode_frame(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits);
    printf("rc=%d, bits=%ld\n", rc, bits);
    
    printf("Recon: ");
    for(int i = 0; i < np; i++) printf("%d ", recon[i]);
    printf("\n");
    
    return 0;
}
