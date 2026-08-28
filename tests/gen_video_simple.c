/* gen_video_simple.c: Simple video example generator */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"

#define W 320
#define H 240
#define N 30

int main(void) {
    const int np = W * H;
    uint8_t* frames = malloc((size_t)N * np);
    uint8_t* recons = malloc((size_t)N * np);
    uint8_t* recon_buf = malloc(np);
    
    /* Generate simple test pattern */
    printf("Generating %d frames at %dx%d...\n", N, W, H);
    for(int f = 0; f < N; f++) {
        for(int i = 0; i < np; i++) {
            frames[f * np + i] = (uint8_t)((i + f * 7) % 256);
        }
    }
    
    /* Save original */
    printf("Saving original...\n");
    FILE* fp = fopen("docs/video_examples/frames/original.raw", "wb");
    for(int f = 0; f < N; f++) fwrite(frames + f * np, 1, np, fp);
    fclose(fp);
    
    /* Encode at QP=26, fast preset */
    printf("Encoding at QP=26, fast preset...\n");
    fp = fopen("docs/video_examples/encoded/qp26_fast.raw", "wb");
    for(int f = 0; f < N; f++) {
        long bits = 0;
        int ret = wubu_encode_frame_preset(frames + f * np, 
                                  f > 0 ? recons + (f-1) * np : NULL,
                                  NULL, W, H, 26,
                                  f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME,
                                  NULL, recon_buf, &bits, 2);
        printf("  Frame %d: ret=%d bits=%ld\n", f, ret, bits);
        fwrite(recon_buf, 1, np, fp);
        memcpy(recons + f * np, recon_buf, np);
    }
    fclose(fp);
    
    /* Create A/B comparison */
    printf("Creating A/B comparison...\n");
    int W2 = W * 2;
    uint8_t* side = malloc((size_t)W2 * H);
    fp = fopen("docs/video_examples/encoded/ab_comparison.raw", "wb");
    for(int f = 0; f < N; f++) {
        memset(side, 0, (size_t)W2 * H);
        for(int y = 0; y < H; y++) {
            memcpy(side + y * W2, frames + f * np + y * W, W);
            memcpy(side + y * W2 + W, recons + f * np + y * W, W);
        }
        fwrite(side, 1, (size_t)W2 * H, fp);
    }
    fclose(fp);
    
    free(frames);
    free(recons);
    free(recon_buf);
    free(side);
    
    printf("Done!\n");
    return 0;
}