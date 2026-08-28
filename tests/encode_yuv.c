/* encode_yuv.c: Simple YUV encoder */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"

int main(int argc, char** argv) {
    if(argc < 7) {
        printf("Usage: %s input.yuv output.yuv W H N qp preset\n", argv[0]);
        return 1;
    }
    
    const char* input = argv[1];
    const char* output = argv[2];
    int W = atoi(argv[3]);
    int H = atoi(argv[4]);
    int N = atoi(argv[5]);
    int qp = atoi(argv[6]);
    int preset = atoi(argv[7]);
    
    int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    uint8_t* recons = calloc((size_t)N * np, 1);
    
    FILE* fin = fopen(input, "rb");
    FILE* fout = fopen(output, "wb");
    
    if(!fin || !fout) {
        printf("Cannot open files\n");
        return 1;
    }
    
    printf("Encoding %d frames at %dx%d, QP=%d, preset=%d...\n", N, W, H, qp, preset);
    
    for(int f = 0; f < N; f++) {
        if(fread(frame, 1, np, fin) != (size_t)np) {
            printf("Read error at frame %d\n", f);
            break;
        }
        
        long bits = 0;
        wubu_encode_frame_preset(frame, 
                                  f > 0 ? recons + (f-1) * np : NULL,
                                  NULL, W, H, qp,
                                  f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME,
                                  NULL, recon, &bits, preset);
        
        fwrite(recon, 1, np, fout);
        memcpy(recons + f * np, recon, np);
        
        if(f % 10 == 0) printf("  Frame %d: %ld bits\n", f, bits);
    }
    
    fclose(fin);
    fclose(fout);
    free(frame);
    free(recon);
    free(recons);
    
    printf("Done!\n");
    return 0;
}