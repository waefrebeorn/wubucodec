
/* raw_encode.c: Encode raw YUV using WuBuCodec */
#include <stdio.h>
#include <stdlib.h>
#include "wubu_encoder.h"

int main(int argc, char** argv) {
    if(argc < 4) {
        printf("Usage: %s input.raw output.raw W H [qp] [preset]\n", argv[0]);
        return 1;
    }
    
    const char* input = argv[1];
    const char* output = argv[2];
    int W = atoi(argv[3]);
    int H = atoi(argv[4]);
    int qp = (argc > 5) ? atoi(argv[5]) : 26;
    int preset = (argc > 6) ? atoi(argv[6]) : 2;
    
    int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    uint8_t* prev = NULL;
    
    FILE* fin = fopen(input, "rb");
    FILE* fout = fopen(output, "wb");
    
    if(!fin || !fout) {
        printf("Cannot open files\n");
        return 1;
    }
    
    int frame_idx = 0;
    while(fread(frame, 1, np, fin) == (size_t)np) {
        long bits = 0;
        int frame_type = (frame_idx == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        
        int ret = wubu_encode_frame_preset(frame, prev, NULL, W, H, qp, frame_type, NULL, recon, &bits, preset);
        if(ret != 0) {
            printf("Encode failed at frame %d: %d\n", frame_idx, ret);
            break;
        }
        
        fwrite(recon, 1, np, fout);
        
        if(!prev) prev = malloc(np);
        memcpy(prev, recon, np);
        
        frame_idx++;
    }
    
    printf("Encoded %d frames\n", frame_idx);
    
    free(frame);
    free(recon);
    free(prev);
    fclose(fin);
    fclose(fout);
    
    return 0;
}
