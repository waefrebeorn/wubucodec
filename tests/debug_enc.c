/* debug_enc.c: Debug encoder on real content */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"

#define W 320
#define H 240

int main() {
    printf("=== Debug Encoder ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    
    /* Test 1: All zeros */
    printf("Test 1: All zeros...\n");
    memset(frame, 0, np);
    long bits = 0;
    int rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
    printf("  rc=%d, bits=%ld\n", rc, bits);
    
    /* Test 2: All 255 */
    printf("Test 2: All 255...\n");
    memset(frame, 255, np);
    bits = 0;
    rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
    printf("  rc=%d, bits=%ld\n", rc, bits);
    
    /* Test 3: Random noise */
    printf("Test 3: Random noise...\n");
    for(int i = 0; i < np; i++) frame[i] = rand() % 256;
    bits = 0;
    rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
    printf("  rc=%d, bits=%ld\n", rc, bits);
    
    /* Test 4: Gradient */
    printf("Test 4: Gradient...\n");
    for(int i = 0; i < np; i++) frame[i] = (uint8_t)((i * 255) / np);
    bits = 0;
    rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
    printf("  rc=%d, bits=%ld\n", rc, bits);
    
    /* Test 5: Real cartoon frame */
    printf("Test 5: Real cartoon frame...\n");
    FILE* f = fopen("/tmp/balloon.raw", "rb");
    if(f) {
        if(fread(frame, 1, np, f) == (size_t)np) {
            bits = 0;
            rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
            printf("  rc=%d, bits=%ld\n", rc, bits);
        } else {
            printf("  Short read\n");
        }
        fclose(f);
    } else {
        printf("  Cannot open file\n");
    }
    
    /* Test 6: Two frames (P-frame) */
    printf("Test 6: Two frames (I then P)...\n");
    f = fopen("/tmp/balloon.raw", "rb");
    if(f) {
        if(fread(frame, 1, np, f) == (size_t)np) {
            bits = 0;
            rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
            printf("  I-frame: rc=%d, bits=%ld\n", rc, bits);
            
            if(rc == 0) {
                uint8_t* prev = malloc(np);
                memcpy(prev, recon, np);
                
                if(fread(frame, 1, np, f) == (size_t)np) {
                    bits = 0;
                    rc = wubu_encode_frame_preset(frame, prev, NULL, W, H, 26, WUBU_P_FRAME, NULL, recon, &bits, 2);
                    printf("  P-frame: rc=%d, bits=%ld\n", rc, bits);
                }
                free(prev);
            }
        }
        fclose(f);
    }
    
    free(frame);
    free(recon);
    
    printf("=== Done ===\n");
    return 0;
}