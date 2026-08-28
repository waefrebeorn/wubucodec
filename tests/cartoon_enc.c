/* cartoon_enc.c: Encode cartoon in chunks to avoid memory issues */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"

#define W 320
#define H 240
#define CHUNK_SIZE 100

int main() {
    printf("=== Cartoon Encoder ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    uint8_t* prev = malloc(np);
    int first = 1;
    
    FILE* f = fopen("/tmp/balloon.raw", "rb");
    if(!f) { printf("Cannot open\n"); return 1; }
    
    int total_frames = 0;
    int chunk_num = 0;
    
    while(!feof(f)) {
        char fname[256];
        snprintf(fname, sizeof(fname), "/tmp/balloon_chunk_%03d.raw", chunk_num);
        FILE* fout = fopen(fname, "wb");
        if(!fout) { printf("Cannot create %s\n", fname); break; }
        
        int chunk_frames = 0;
        while(chunk_frames < CHUNK_SIZE && fread(frame, 1, np, f) == (size_t)np) {
            long bits = 0;
            int frame_type = first ? WUBU_I_FRAME : WUBU_P_FRAME;
            int rc = wubu_encode_frame_preset(frame, first ? NULL : prev, NULL, W, H, 26, frame_type, NULL, recon, &bits, 2);
            
            if(rc != 0) {
                printf("Encode failed at frame %d: %d\n", total_frames, rc);
                break;
            }
            
            fwrite(recon, 1, np, fout);
            memcpy(prev, recon, np);
            first = 0;
            chunk_frames++;
            total_frames++;
        }
        
        fclose(fout);
        printf("Chunk %d: %d frames (total: %d)\n", chunk_num, chunk_frames, total_frames);
        chunk_num++;
        
        if(chunk_frames < CHUNK_SIZE) break;
    }
    
    fclose(f);
    free(frame);
    free(recon);
    free(prev);
    
    printf("Total: %d frames in %d chunks\n", total_frames, chunk_num);
    return 0;
}