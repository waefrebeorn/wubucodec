/* wubu_vbs_bench.c — benchmark variable block sizes for 8K */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_encoder.h"

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

#define W 1920
#define H 1080
#define NF 12

int main(void){
    const int np = W*H;
    uint8_t* frames = malloc((size_t)NF * np);
    
    for(int f=0; f<NF; f++){
        for(int y=0; y<H; y++){
            for(int x=0; x<W; x++){
                float edge = 960.0f + f * 0.7f;
                float v = (x < edge) ? 100.0f : 200.0f;
                v += 20.0f * sinf(2.0f * M_PI * (x + f) / W) * cosf(2.0f * M_PI * y / H);
                if(v<0)v=0; if(v>255)v=255;
                frames[(size_t)f*np + y*W + x] = (uint8_t)v;
            }
        }
    }
    
    printf("=== WuBuCodec Variable Block Size Benchmark (1920x1080) ===\n\n");
    
    /* Standard 8x8 blocks */
    {
        int qp = 26;
        uint8_t* recon = malloc(np);
        uint8_t* ref = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        long bits;
        
        wubu_encode_frame(frames, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, ref, &bits);
        total_bits = bits;
        
        for(int f=1; f<NF; f++){
            bits = 0;
            wubu_encode_frame(frames + f*np, ref, NULL, W, H, qp, WUBU_P_FRAME, NULL, recon, &bits);
            total_bits += bits;
            memcpy(ref, recon, np);
        }
        
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        
        printf("8x8 blocks: %6.1f fps, %7.1f kbps\n", fps, total_bits * 30.0 / NF / 1024.0);
        free(recon);
        free(ref);
    }
    
    /* With variable block sizes (via preset 0 = superfast which uses larger blocks) */
    {
        int qp = 26;
        uint8_t* recon = malloc(np);
        uint8_t* ref = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        long bits;
        
        wubu_encode_frame_preset(frames, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, ref, &bits, 0);
        total_bits = bits;
        
        for(int f=1; f<NF; f++){
            bits = 0;
            wubu_encode_frame_preset(frames + f*np, ref, NULL, W, H, qp, WUBU_P_FRAME, NULL, recon, &bits, 0);
            total_bits += bits;
            memcpy(ref, recon, np);
        }
        
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        
        printf("Variable bs: %6.1f fps, %7.1f kbps\n", fps, total_bits * 30.0 / NF / 1024.0);
        free(recon);
        free(ref);
    }
    
    free(frames);
    return 0;
}
