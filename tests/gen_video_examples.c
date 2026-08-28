/* gen_video_examples.c: Generate test content, encode, save YUV */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_encoder.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define W 640
#define H 480
#define N 10  /* fewer frames for debugging */

int main(void) {
    const int np = W * H;
    uint8_t* frames = calloc((size_t)N * np, 1);
    uint8_t* recons = calloc((size_t)N * np, 1);
    uint8_t* recon_buf = calloc(np, 1);
    
    /* Generate frames */
    printf("Generating %d frames at %dx%d...\n", N, W, H);
    for(int f = 0; f < N; f++) {
        for(int y = 0; y < H; y++) {
            for(int x = 0; x < W; x++) {
                double v = 128.0;
                int edge = 200 + f * 3;
                if(abs(x - edge) < 2) v = 255.0;
                else if(x < edge) v = 100.0;
                else v = 200.0;
                v += 20.0 * sin(2.0 * M_PI * 8.0 * x / W) * cos(2.0 * M_PI * 8.0 * y / H);
                int cx = 320 + (int)(100 * cos(2.0 * M_PI * f / N));
                int cy = 240 + (int)(80 * sin(2.0 * M_PI * f / N));
                if((x-cx)*(x-cx) + (y-cy)*(y-cy) < 1600) v = 255.0;
                if(abs(y - (100 + f * 2)) < 5) v = 0.0;
                if(v < 0) v = 0;
                if(v > 255) v = 255;
                frames[f * np + y * W + x] = (uint8_t)v;
            }
        }
    }
    
    /* Save original */
    printf("Saving original.yuv...\n");
    FILE* fp = fopen("docs/video_examples/frames/original.yuv", "wb");
    if(!fp) { printf("Cannot open\n"); return 1; }
    for(int f = 0; f < N; f++) fwrite(frames + f * np, 1, np, fp);
    fclose(fp);
    
    /* Encode one frame at a time */
    printf("Encoding...\n");
    fp = fopen("docs/video_examples/encoded/qp26_fast.yuv", "wb");
    if(!fp) { printf("Cannot open\n"); return 1; }
    
    for(int f = 0; f < N; f++) {
        printf("  Frame %d...\n", f);
        long bits = 0;
        int ret = wubu_encode_frame_preset(frames + f * np, 
                                  f > 0 ? recons + (f-1) * np : NULL,
                                  NULL, W, H, 26,
                                  f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME,
                                  NULL, recon_buf, &bits, 2);
        if(ret != 0) {
            printf("  Encode failed: %d\n", ret);
            break;
        }
        fwrite(recon_buf, 1, np, fp);
        memcpy(recons + f * np, recon_buf, np);
    }
    fclose(fp);
    
    /* Create A/B comparison */
    printf("Creating A/B comparison...\n");
    int W2 = W * 2;
    uint8_t* side = malloc((size_t)W2 * H);
    fp = fopen("docs/video_examples/encoded/ab_comparison.yuv", "wb");
    if(!fp) { printf("Cannot open\n"); return 1; }
    
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